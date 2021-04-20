#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import random
import numpy as np

from pxr import Gf, UsdGeom
from enum import Enum
import omni
import carb

from omni.physx.scripts.physicsUtils import add_ground_plane
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.utils._isaac_utils import math as math_utils
from omni.isaac.samples.scripts.utils.world import World
# from omni.isaac.samples.scripts.utils.franka import Franka, default_config
from .franka import Franka, default_config

from .scenario import set_translate, set_rotate, Scenario, setup_physics
from copy import copy

from omni.physx import _physx
from collections import deque


def normalize(a):
    norm = np.linalg.norm(a)
    return a / norm

def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")  # create an empty Xform at the given path
    envPrim.GetReferences().AddReference(prim_usd_path)  # attach the USD to the given path
    set_translate(envPrim, location)  # set pose

class SM_events(Enum):
    START = 0
    WAYPOINT_REACHED = 1
    GOAL_REACHED = 2
    ATTACHED = 3
    DETACHED = 4
    TIMEOUT = 5
    STOP = 6
    NONE = 7  # no event ocurred, just clocks


class SM_states(Enum):
    STANDBY = 0  # Default state, does nothing unless enters with event START
    PICKING = 1
    ATTACH = 2
    HOLDING = 3
    GRASPING = 4
    LIFTING = 5


statedic = {0: "orig", 1: "axis_x", 2: "axis_y", 3: "axis_z"}


class PickAndPlaceStateMachine(object):
    """
    Self-contained state machine class for Robot Behavior. Each machine state may react to different events,
    and the handlers are defined as in-class functions
    """

    def __init__(self, stage, robot, ee_prim, target_body, default_position):
        self.robot = robot
        self.dc = robot.dc
        self.end_effector = ee_prim
        self.end_effector_handle = None
        self._stage = stage
        self.current = target_body

        self.start_time = 0.0
        self.start = False
        self._time = 0.0
        self.default_timeout = 0.5
        self.default_position = copy(default_position)
        self.target_position = default_position
        self.reset = False
        self.waypoints = deque()
        self.thresh = {}
        # Threshold to clear waypoints/goal
        # (any waypoint that is not final will be cleared with the least precision)
        self.precision_thresh = [
            [0.0005, 0.0025, 0.0025, 0.0025],
            [0.0005, 0.005, 0.005, 0.005],
            [0.05, 0.2, 0.2, 0.2],
            [0.08, 0.4, 0.4, 0.4],
            [0.18, 0.6, 0.6, 0.6],
        ]
        self.add_bin = None

        # Event management variables

        # Used to verify if the goal was reached due to robot moving or it had never left previous target
        self._is_moving = False
        self._attached = False  # Used to flag the Attached/Detached events on a change of state from the end effector
        self._detached = False

        self.is_closed = False
        self.pick_count = 0
        # Define the state machine handling functions
        self.sm = {}
        # Make empty state machine for all events and states
        for s in SM_states:
            self.sm[s] = {}
            for e in SM_events:
                self.sm[s][e] = self._empty
                self.thresh[s] = 0

        # Fill in the functions to handle each event for each status
        self.sm[SM_states.STANDBY][SM_events.START] = self._standby_start
        self.sm[SM_states.STANDBY][SM_events.GOAL_REACHED] = self._standby_goal_reached
        self.thresh[SM_states.STANDBY] = 3

        self.sm[SM_states.PICKING][SM_events.GOAL_REACHED] = self._picking_goal_reached
        # self.sm[SM_states.PICKING][SM_events.NONE] = self._picking_no_event
        self.thresh[SM_states.PICKING] = 1

        self.sm[SM_states.GRASPING][SM_events.ATTACHED] = self._grasping_attached
        # self.sm[SM_states.ATTACH][SM_events.GOAL_REACHED] = self._attach_goal_reached
        # self.sm[SM_states.ATTACH][SM_events.ATTACHED] = self._attach_attached

        self.sm[SM_states.LIFTING][SM_events.GOAL_REACHED] = self._lifting_goal_reached
        # self.sm[SM_states.HOLDING][SM_events.GOAL_REACHED] = self._holding_goal_reached
        # self.thresh[SM_states.HOLDING] = 3
        for s in SM_states:
            self.sm[s][SM_events.DETACHED] = self._all_detached

        self.current_state = SM_states.STANDBY
        self.previous_state = -1
        self._physxIFace = _physx.acquire_physx_interface()

    # Auxiliary functions

    def _empty(self, *args):
        """
        Empty function to use on states that do not react to some specific event
        """
        pass

    def change_state(self, new_state):
        """
        Function called every time a event handling changes current state
        """
        self.current_state = new_state
        self.start_time = self._time
        carb.log_warn(str(new_state))

    def goalReached(self):
        """
        Checks if the robot has reached a certain waypoint in the trajectory
        """
        if self._is_moving:
            state = self.robot.end_effector.status.current_frame
            target = self.robot.end_effector.status.current_target
            error = 0
            for i in [0, 2, 3]:
                k = statedic[i]
                state_v = state[k]
                target_v = target[k]
                error = np.linalg.norm(state_v - target_v)
                # General Threshold is the least strict
                thresh = self.precision_thresh[-1][i]
                # NOTE: use the below as a method to identify when the goal point was reached 
                # if the target is a goal point, use the defined threshold for the current state
                if len(self.waypoints) == 0:
                    thresh = self.precision_thresh[self.thresh[self.current_state]][i]
                # carb.log_warn(f'ERROR: {error}, THRESHOLD: {thresh}')
                if error > thresh:
                    return False
            self._is_moving = False
            return True
        return False

    def get_current_state_tr(self):
        """
        Gets current End Effector Transform, converted from Motion position and Rotation matrix
        """
        # Gets end effector frame
        state = self.robot.end_effector.status.current_frame

        orig = state["orig"] * 100.0

        mat = Gf.Matrix3f(
            *state["axis_x"].astype(float), *state["axis_y"].astype(float), *state["axis_z"].astype(float)
        )
        q = mat.ExtractRotation().GetQuaternion()
        (q_x, q_y, q_z) = q.GetImaginary()
        q = [q_x, q_y, q_z, q.GetReal()]
        tr = _dynamic_control.Transform()
        tr.p = list(orig)
        tr.r = q
        return tr

    def ray_cast(self, x_offset=0.15, y_offset=3.0, z_offset=0.0):
        """
        Projects a raycast forward from the end effector, with an offset in end effector space defined by (x_offset, y_offset, z_offset)
        if a hit is found on a distance of 100 centimiters, returns the object usd path and its distance
        """
        tr = self.get_current_state_tr()
        offset = _dynamic_control.Transform()
        offset.p = (x_offset, y_offset, z_offset)
        raycast_tf = math_utils.mul(tr, offset)
        origin = raycast_tf.p
        rayDir = math_utils.get_basis_vector_x(raycast_tf.r)
        hit = self._physxIFace.raycast_closest(origin, rayDir, 100.0)
        if hit["hit"]:
            usdGeom = UsdGeom.Mesh.Get(self._stage, hit["rigidBody"])
            distance = hit["distance"]
            return usdGeom.GetPath().pathString, distance
        return None, 10000.0

    def lerp_to_pose(self, pose, n_waypoints=1):
        """
        adds spherical linear interpolated waypoints from last pose in the waypoint list to the provided pose
        if the waypoit list is empty, use current pose
        """
        if len(self.waypoints) == 0:
            start = self.get_current_state_tr()
            start.p = math_utils.mul(start.p, 0.01)
        else:
            start = self.waypoints[-1]

        if n_waypoints > 1:
            for i in range(n_waypoints):
                self.waypoints.append(math_utils.slerp(start, pose, (i + 1.0) / n_waypoints))
        else:
            self.waypoints.append(pose)

    def move_to_zero(self):
        self._is_moving = False
        self.robot.end_effector.go_local(
            orig=[], axis_x=[], axis_y=[], axis_z=[], use_default_config=True, wait_for_target=False, wait_time=5.0
        )

    def move_to_target(self):
        xform_attr = self.target_position
        self._is_moving = True

        orig = np.array([xform_attr.p.x, xform_attr.p.y, xform_attr.p.z])
        axis_y = np.array(math_utils.get_basis_vector_y(xform_attr.r))
        axis_z = np.array(math_utils.get_basis_vector_z(xform_attr.r))
        self.robot.end_effector.go_local(
            orig=orig,
            axis_x=[],
            axis_y=axis_y,
            axis_z=axis_z,
            use_default_config=True,
            wait_for_target=False,
            wait_time=5.0,
        )

    # TODO: Add grasp angle
    def get_target_to_object(self, offset_position=[]):
        """
        Gets target pose to end effector on a given target, with an offset on the end effector actuator direction given
        by [offset_up, offset_down]
        """
        offset = _dynamic_control.Transform()
        if offset_position:
            
            offset.p.x = offset_position[0]
            offset.p.y = offset_position[1]
            offset.p.z = offset_position[2]

        body_handle = self.dc.get_rigid_body(self.current)
        obj_pose = self.dc.get_rigid_body_pose(body_handle)
        target_pose = _dynamic_control.Transform()
        target_pose.p = obj_pose.p
        target_pose.r = [0.0, 1.0, 0.0, 0.0]
        target_pose = math_utils.mul(target_pose, offset)
        target_pose.p = math_utils.mul(target_pose.p, 0.01)
        return target_pose

    def set_target_to_object(self, offset_position=[], n_waypoints=1, clear_waypoints=True):
        """
        Clears waypoints list, and sets a new waypoint list towards the target pose for an object.
        """
        target_position = self.get_target_to_object(offset_position=offset_position)
        # linear interpolate to target pose
        if clear_waypoints:
            self.waypoints.clear()
        self.lerp_to_pose(target_position, n_waypoints=n_waypoints)
        # Get first waypoint target
        self.target_position = self.waypoints.popleft()

    def step(self, timestamp, start=False, reset=False):
        """
            Steps the State machine, handling which event to call
        """
        if self.current_state != self.previous_state:
            self.previous_state = self.current_state
        if not self.start:
            self.start = start

        # NOTE: This may be a good way to evaluate whether the graps was a success or failure
        finger_velocity = self.robot.end_effector.gripper.get_velocity(from_articulation=True)
        # carb.log_warn(f'WIDTH: {self.robot.end_effector.gripper.width:.4f}, ACTUAL WIDTH: {self.robot.end_effector.gripper.get_width():.4f}, FINGER_VELOCITY: ({finger_velocity[0]:.4f}, {finger_velocity[1]:.4f}), HISTORY_STD: {np.array(self.robot.end_effector.gripper.width_history).std():.2e}')
        # if self.is_closed and (self.current_state == SM_states.GRASPING or self.current_state == SM_states.LIFTING):
        if self.current_state == SM_states.GRASPING or self.current_state == SM_states.LIFTING:
            # object grasped
            if not self.robot.end_effector.gripper.is_closed(1e-1) and not self.robot.end_effector.gripper.is_moving(1e-2):
                self._attached = True
                # self.is_closed = False
            # object not grasped
            elif self.robot.end_effector.gripper.is_closed(1e-1):
                self._detached = True
                self.is_closed = True

        # Process events
        if reset:
            # reset to default pose, clear waypoints, and re-initialize event handlers
            self.current_state = SM_states.STANDBY
            self.robot.end_effector.gripper.open()
            self.start = False
            self.waypoints.clear()
            self.target_position = self.default_position
            self.move_to_target()
        elif self._detached:
            self._detached = False
            self.sm[self.current_state][SM_events.DETACHED]()
        elif self.goalReached():
            if len(self.waypoints) == 0:
                self.sm[self.current_state][SM_events.GOAL_REACHED]()
            else:
                self.target_position = self.waypoints.popleft()
                self.move_to_target()
                self.start_time = self._time
        elif self.current_state == SM_states.STANDBY and self.start:
            self.sm[self.current_state][SM_events.START]()
        elif self._attached:
            self._attached = False
            self.sm[self.current_state][SM_events.ATTACHED]()
        elif self._time - self.start_time > self.default_timeout:
            self.sm[self.current_state][SM_events.TIMEOUT]()
        else:
            self.sm[self.current_state][SM_events.NONE]()

    # Event handling functions. Each state has its own event handler function depending on which event happened

    def _standby_start(self, *args):
        """
        Handles the start event when in standby mode.
        Proceeds to pick up the next bin on the queue, and set the arm
        to move towards the bin from current  position.
        switches to picking state.
        """
        # Tell motion planner controller to ignore current object as an obstacle
        self.pick_count = 0
        self.lerp_to_pose(self.default_position, 1)
        self.lerp_to_pose(self.default_position, 60)
        self.robot.end_effector.gripper.open()
        # set target above the current bin with offset of 20 cm
        self.set_target_to_object(offset_position=[0.0, 0.0, -10.0], n_waypoints=90, clear_waypoints=False)
        # TODO: add another command to lower arm towards the object
        self.lerp_to_pose(self.waypoints[-1], 90)
        self.set_target_to_object(offset_position=[0.0, 0.0, -2.0], n_waypoints=90, clear_waypoints=False)
        # start arm movement
        self.move_to_target()
        # Move to next state
        self.change_state(SM_states.PICKING)

    # NOTE: As is, this method is never executed
    def _standby_goal_reached(self, *args):
         """
         Finished processing a bin, moves up the stack position for next bin placement
         """
         self.move_to_zero()
         self.start = True

    # def _attach_goal_reached(self, *args):
    #     """
    #     Handles a state machine step when the target goal is reached, and the machine is on attach state
    #     """
    #     self.robot.end_effector.gripper.close()
    #     self.lerp_to_pose(self.target_position, 60)  # Wait 1 second in place for attachment
    #     if self.robot.end_effector.gripper.is_closed():
    #         self._attached = True
    #         self.is_closed = True
    #     else:  # Failed to attach so return grasp to try again
    #         # move up 25 centimiters and return to picking state
    #         offset = _dynamic_control.Transform()
    #         offset.p = (-0.25, 0.0, 0.0)
    #         self.target_position = math_utils.mul(self.target_position, offset)
    #         self.move_to_target()
    #         self.change_state(SM_states.PICKING)

    # def _attach_attached(self, *args):
    #     """
    #     Handles a state machine step when the target goal is reached, and the machine is on attach state
    #     """
    #     self.waypoints.clear()
    #     target_position = _dynamic_control.Transform()
    #     target_position.p = [0.0, 0.81, 0.58]
    #     target_position.r = [0, -1, 0, 0]
    #     print(target_position.r)
    #     self.lerp_to_pose(target_position, 360)
    #     self.target_position = self.waypoints.popleft()
    #     self.move_to_target()
    #     self.change_state(SM_states.HOLDING)

    def _picking_goal_reached(self, *args):
        """
        Handles a state machine step when goal was reached event happens, while on picking state
        ensures the bin obstacle is suppressed for the planner, Updates the target position
        to where the bin surface is, and send the robot to move towards it. No change of state happens
        """
        # obj, distance = self.ray_cast()
        # if obj is not None:
        #     # Set target towards surface of the bin
        #     tr = self.get_current_state_tr()
        #     offset = _dynamic_control.Transform()
        #     offset.p = (distance + 0.15, 0, 0)

        #     target = math_utils.mul(tr, offset)
        #     target.p = math_utils.mul(target.p, 0.01)
        #     offset.p.x = -0.05

        #     pre_target = math_utils.mul(target, offset)
        #     self.lerp_to_pose(pre_target, n_waypoints=40)
        #     self.lerp_to_pose(target, n_waypoints=30)
        #     self.lerp_to_pose(target, n_waypoints=30)
        #     self.target_position = self.waypoints.popleft()
        #     self.move_to_target()
        #     self.change_state(SM_states.ATTACH)
        self.robot.end_effector.gripper.close()
        self.is_closed = True
        # Move to next state
        self.move_to_target()
        self.robot.end_effector.gripper.width_history.clear()
        self.change_state(SM_states.GRASPING)

    def _picking_no_event(self, *args):
        """
        Handles a state machine step when no event happened, while on picking state
        ensures the bin obstacle is suppressed for the planner, Updates the target position
        to where the bin is, and send the robot to move towards it. No change of state happens
        """
        # self.set_target_to_object(offset_position=[0.0, 0.0, -2.0], n_waypoints=10, clear_waypoints=True)
        # self.move_to_target()
        pass

    def _grasping_attached(self, *args):
        self.waypoints.clear()
        offset = _dynamic_control.Transform()
        offset.p.z = -10
        target_pose = math_utils.mul(self.get_current_state_tr(), offset)
        target_pose.p = math_utils.mul(target_pose.p, 0.01)
        self.lerp_to_pose(target_pose, n_waypoints=60)
        self.lerp_to_pose(target_pose, n_waypoints=90)
        # Move to next state
        self.move_to_target()
        self.robot.end_effector.gripper.width_history.clear()
        self.change_state(SM_states.LIFTING)

    def _grasping_goal_reached(self, *args):
        # self.robot.end_effector.gripper.open()
        # self.is_closed = False
        # # Move to next state
        # self.move_to_target()
        # self.change_state(SM_states.STANDBY)
        pass

    def _lifting_goal_reached(self, *args):
        self.is_closed = False
        self.robot.end_effector.gripper.open()
        self._all_detached()
        carb.log_warn('GRASP SUCCESSFUL')

    # def _holding_goal_reached(self, *args):

    #     if self.add_bin is not None:
    #         self.add_bin()
    #     self.lerp_to_pose(self.target_position, 20)
    #     self.move_to_target()

    def _all_detached(self, *args):
         self.current_state = SM_states.STANDBY
         self.start = False
         self.waypoints.clear()
         self.lerp_to_pose(self.target_position, 60)
         self.lerp_to_pose(self.default_position, 10)
         self.lerp_to_pose(self.default_position, 60)
         self.move_to_target()
         carb.log_warn('GRASP UNSUCCESSFUL')


class GraspObject(Scenario):
    """ Defines an obstacle avoidance scenario

    Scenarios define the life cycle within kit and handle init, startup, shutdown etc.
    """

    def __init__(self, editor, dc, mp):
        super().__init__(editor, dc, mp)
        self._paused = True
        self._start = False
        self._reset = False
        self._time = 0
        self._start_time = 0
        self.current_state = SM_states.STANDBY
        self.timeout_max = 8.0
        self.pick_and_place = None
        self._pending_disable = False
        self._pending_stop = False
        self._gripper_open = False

        self.current_obj = 0
        self.max_objs = 100
        self.num_objs = 3

        self.add_objects_timeout = -1
        self.franka_solid = None

    def __del__(self):
        if self.franka_solid:
            self.franka_solid.end_effector.gripper = None
        super().__del__()

    def on_startup(self):
        super().on_startup()

    def create_franka(self, *args):
        super().create_franka()
        if self.asset_path is None:
            return

        self.objects = [
            self.asset_path + "/Props/Flip_Stack/large_corner_bracket_physics.usd",
            self.asset_path + "/Props/Flip_Stack/screw_95_physics.usd",
            self.asset_path + "/Props/Flip_Stack/screw_99_physics.usd",
            self.asset_path + "/Props/Flip_Stack/small_corner_bracket_physics.usd",
            self.asset_path + "/Props/Flip_Stack/t_connector_physics.usd",
        ]
        self.current_obj = 0

        # Load robot environment and set its transform
        self.env_path = "/scene"
        robot_usd = self.asset_path + "/Robots/Franka/franka.usd"
        robot_path = "/scene/robot"
        create_prim_from_usd(self._stage, robot_path, robot_usd, Gf.Vec3d(0, 0, 0))

        # Set robot end effector Target
        target_path = "/scene/target"
        if self._stage.GetPrimAtPath(target_path):
            return

        GoalPrim = self._stage.DefinePrim(target_path, "Xform")
        self.default_position = _dynamic_control.Transform()
        self.default_position.p = [0.4, 0.0, 0.3]
        self.default_position.r = [0.0, 1.0, 0.0, 0.0] #TODO: Check values for stability
        p = self.default_position.p
        r = self.default_position.r
        set_translate(GoalPrim, Gf.Vec3d(p.x * 100, p.y * 100, p.z * 100))
        set_rotate(GoalPrim, Gf.Matrix3d(Gf.Quatd(r.w, r.x, r.y, r.z)))

        # Setup physics simulation
        add_ground_plane(self._stage, "/groundPlane", "Z", 1000.0, Gf.Vec3f(0.0), Gf.Vec3f(1.0))
        setup_physics(self._stage)
        self.add_bin()

    def add_bin(self, *args):
        self.create_new_objects(args)

    def create_new_objects(self, *args):
        prim_usd_path = self.objects[random.randint(0, len(self.objects) - 1)]
        prim_env_path = "/scene/objects/object_{}".format(self.current_obj)
        location = Gf.Vec3d(40, 2 * self.current_obj, 10)
        create_prim_from_usd(self._stage, prim_env_path, prim_usd_path, location)
        self.current_obj += 1

    def register_assets(self, *args):

        # Prim path of two blocks and their handles
        object_path = "/scene/objects/object_0"
        object_children = self._stage.GetPrimAtPath(object_path).GetChildren()
        for child in object_children:
            child_path = child.GetPath().pathString
            body_handle = self._dc.get_rigid_body(child_path)
            if body_handle != 0:
                self.bin_path = child_path

        ## register world with RMP
        self.world =  World(self._dc, self._mp)

        ## register robot with RMP
        robot_path = "/scene/robot"
        self.franka_solid = Franka(
            self._stage, self._stage.GetPrimAtPath(robot_path), self._dc, self._mp, self.world, default_config
        )

        # TODO: register objects

        # register stage machine 
        self.pick_and_place = PickAndPlaceStateMachine(
            self._stage,
            self.franka_solid,
            self._stage.GetPrimAtPath("/scene/robot/panda_hand"),
            self.bin_path,
            self.default_position,
        )

    def perform_tasks(self, *args):
        self._start = True
        self._paused = False
        return False

    # TODO: update method
    def step(self, step):
        if self._editor.is_playing():
            if self._pending_stop:
                self.stop_tasks()
                return
            # Updates current references and locations for the robot.
            self.world.update()
            self.franka_solid.update()

            target = self._stage.GetPrimAtPath("/scene/target")
            xform_attr = target.GetAttribute("xformOp:transform")
            if self._reset:
                self._paused = False
            if not self._paused:
                self._time += 1.0 / 60.0
                self.pick_and_place.step(self._time, self._start, self._reset)
                if self._reset:
                    self._paused = (self._time - self._start_time) < self.timeout_max
                    self._time = 0
                    self._start_time = 0
                    p = self.default_position.p
                    r = self.default_position.r
                    set_translate(target, Gf.Vec3d(p.x * 100, p.y * 100, p.z * 100))
                    set_rotate(target, Gf.Matrix3d(Gf.Quatd(r.w, r.x, r.y, r.z)))
                else:
                    state = self.franka_solid.end_effector.status.current_target
                    state_1 = self.pick_and_place.target_position
                    tr = state["orig"] * 100.0
                    set_translate(target, Gf.Vec3d(tr[0], tr[1], tr[2]))
                    set_rotate(target, Gf.Matrix3d(Gf.Quatd(state_1.r.w, state_1.r.x, state_1.r.y, state_1.r.z)))
                self._start = False
                self._reset = False
                if self.add_objects_timeout > 0:
                    self.add_objects_timeout -= 1
                    if self.add_objects_timeout == 0:
                        self.create_new_objects()
                # if (
                #     self.pick_and_place.current_state == self.current_state
                #     and self.current_state in [SM_states.PICKING, SM_states.ATTACH]
                #     and (self._time - self._start_time) > self.timeout_max
                # ):
                #     self.stop_tasks()
                # elif self.pick_and_place.current_state != self.current_state:
                #     self._start_time = self._time
                #     print(self._time)
                #     self.current_state = self.pick_and_place.current_state

            if self._paused:
                translate_attr = xform_attr.Get().GetRow3(3)
                rotate_x = xform_attr.Get().GetRow3(0)
                rotate_y = xform_attr.Get().GetRow3(1)
                rotate_z = xform_attr.Get().GetRow3(2)

                orig = np.array(translate_attr) / 100.0
                axis_x = np.array(rotate_x)
                axis_y = np.array(rotate_y)
                axis_z = np.array(rotate_z)
                self.franka_solid.end_effector.go_local(
                    orig=orig,
                    axis_x=axis_x,      # TODO: consider setting this to [] for stability reasons
                    axis_y=axis_y,
                    axis_z=axis_z,
                    use_default_config=True,
                    wait_for_target=False,
                    wait_time=5.0,
                )

    # TODO: update method
    def stop_tasks(self, *args):
        pass

    # TODO: update method
    def pause_tasks(self, *args):
        self._paused = not self._paused
        # if self._paused:
        #     selection = omni.usd.get_context().get_selection()
        #     selection.set_selected_prim_paths(["/scene/target"], False)
        #     target = self._stage.GetPrimAtPath("/scene/target")
        #     xform_attr = target.GetAttribute("xformOp:translate")
        #     translate_attr = np.array(xform_attr.Get().GetRow3(3))
        #     if np.linalg.norm(translate_attr) < 0.01:
        #         p = self.default_position.p
        #         r = self.default_position.r
        #         set_translate(target, Gf.Vec3d(p.x * 100, p.y * 100, p.z * 100))
        #         set_rotate(target, Gf.Matrix3d(Gf.Quatd(r.w, r.x, r.y, r.z)))
        return self._paused

    def open_gripper(self):
        if self._gripper_open:
            self.franka_solid.end_effector.gripper.close()
            self._gripper_open = False
        else:
            self.franka_solid.end_effector.gripper.open()
            self._gripper_open = True
