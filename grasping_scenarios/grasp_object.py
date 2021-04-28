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
from scipy.spatial.transform import Rotation

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
    return stage.GetPrimAtPath(envPrim.GetPath().pathString)

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

class GRASP_eval(Enum):
    FAILURE = 0
    SUCCESS = 1

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
        self.default_timeout = 10
        self.default_position = copy(default_position)
        self.target_position = default_position
        self.target_angle = 0 # grasp angle in degrees
        self.reset = False
        self.evaluation = None
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
        self.thresh[SM_states.PICKING] = 1

        self.sm[SM_states.GRASPING][SM_events.ATTACHED] = self._grasping_attached
        # self.thresh[SM_states.GRASPING] = 3

        self.sm[SM_states.LIFTING][SM_events.GOAL_REACHED] = self._lifting_goal_reached
        # self.thresh[SM_states.LIFTING] = 3

        for s in SM_states:
            self.sm[s][SM_events.DETACHED] = self._all_detached
            self.sm[s][SM_events.TIMEOUT] = self._all_timeout

        self.current_state = SM_states.STANDBY
        self.previous_state = -1
        self._physxIFace = _physx.acquire_physx_interface()

    # Auxiliary functions

    def _empty(self, *args):
        """
        Empty function to use on states that do not react to some specific event
        """
        pass

    def change_state(self, new_state, print_state=True):
        """
        Function called every time a event handling changes current state
        """
        self.current_state = new_state
        self.start_time = self._time
        if print_state: carb.log_warn(str(new_state))

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
        target_pose.r = self.get_target_orientation()
        target_pose = math_utils.mul(target_pose, offset)
        target_pose.p = math_utils.mul(target_pose.p, 0.01)
        return target_pose

    def get_target_orientation(self):
        angle = self.target_angle * np.pi / 180
        mat = Gf.Matrix3f(
            -np.cos(angle), -np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0, 0, 0, -1
        )
        q = mat.ExtractRotation().GetQuaternion()
        (q_x, q_y, q_z) = q.GetImaginary()
        q = [q_x, q_y, q_z, q.GetReal()]
        return q

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
        # finger_velocity = self.robot.end_effector.gripper.get_velocity(from_articulation=True)
        # carb.log_warn(f'WIDTH: {self.robot.end_effector.gripper.width:.4f}, ACTUAL WIDTH: {self.robot.end_effector.gripper.get_width():.4f}, FINGER_VELOCITY: ({finger_velocity[0]:.4f}, {finger_velocity[1]:.4f}), HISTORY_STD: {np.array(self.robot.end_effector.gripper.width_history).std():.2e}')
        # carb.log_warn(f'TIME: {self._time:.4f}, START_TIME: {self.start_time:.4f}, DELTA_TIME: {self._time - self.start_time:.4f}')
        # if self.is_closed and (self.current_state == SM_states.GRASPING or self.current_state == SM_states.LIFTING):
        if self.current_state in [SM_states.GRASPING, SM_states.LIFTING]:
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
            self.previous_state = -1
            self.robot.end_effector.gripper.open()
            self.evaluation = None
            self.start = False
            self._time = 0
            self.start_time = self._time
            self.pick_count = 0
            self.waypoints.clear()
            self._detached = False
            self._attached = False
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
                # self.start_time = self._time
        elif self.current_state == SM_states.STANDBY and self.start:
            self.sm[self.current_state][SM_events.START]()
        elif self._attached:
            self._attached = False
            self.sm[self.current_state][SM_events.ATTACHED]()
        elif self._time - self.start_time > self.default_timeout:
            self.sm[self.current_state][SM_events.TIMEOUT]()
        else:
            self.sm[self.current_state][SM_events.NONE]()
        self._time += 1.0 / 60.0

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
        self.evaluation = None
        self.lerp_to_pose(self.default_position, 1)
        self.lerp_to_pose(self.default_position, 60)
        self.robot.end_effector.gripper.open()
        # set target above the current bin with offset of 10 cm
        self.set_target_to_object(offset_position=[0.0, 0.0, -10.0], n_waypoints=90, clear_waypoints=False)
        # pause before lowering to target object
        self.lerp_to_pose(self.waypoints[-1], 180)
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

    def _picking_goal_reached(self, *args):
        """
        Handles a state machine step when goal was reached event happens, while on picking state
        ensures the bin obstacle is suppressed for the planner, Updates the target position
        to where the bin surface is, and send the robot to move towards it. No change of state happens
        """
        self.robot.end_effector.gripper.close()
        self.is_closed = True
        # Move to next state
        self.move_to_target()
        self.robot.end_effector.gripper.width_history.clear()
        self.change_state(SM_states.GRASPING)

    def _grasping_attached(self, *args):
        self.waypoints.clear()
        offset = _dynamic_control.Transform()
        offset.p.z = -10
        target_pose = math_utils.mul(self.get_current_state_tr(), offset)
        target_pose.p = math_utils.mul(target_pose.p, 0.01)
        self.lerp_to_pose(target_pose, n_waypoints=60)
        self.lerp_to_pose(target_pose, n_waypoints=120)
        # Move to next state
        self.move_to_target()
        self.robot.end_effector.gripper.width_history.clear()
        self.change_state(SM_states.LIFTING)

    def _lifting_goal_reached(self, *args):
        self.is_closed = False
        self.robot.end_effector.gripper.open()
        self._all_detached()
        self.pick_count += 1
        self.evaluation = GRASP_eval.SUCCESS
        carb.log_warn(str(GRASP_eval.SUCCESS))

    def _all_timeout(self, *args):
        self.change_state(SM_states.STANDBY, print_state=False)
        self.robot.end_effector.gripper.open()
        self.start = False
        self.waypoints.clear()
        self.target_position = self.default_position
        # self.lerp_to_pose(self.target_position, 60)
        self.lerp_to_pose(self.default_position, 1)
        self.lerp_to_pose(self.default_position, 10)
        self.lerp_to_pose(self.default_position, 60)
        self.move_to_target()
        self.evaluation = GRASP_eval.FAILURE
        carb.log_warn(str(GRASP_eval.FAILURE))

    def _all_detached(self, *args):
        self.change_state(SM_states.STANDBY, print_state=False)
        self.start = False
        self.waypoints.clear()
        self.lerp_to_pose(self.target_position, 60)
        self.lerp_to_pose(self.default_position, 10)
        self.lerp_to_pose(self.default_position, 60)
        self.move_to_target()
        self.evaluation = GRASP_eval.FAILURE
        carb.log_warn(str(GRASP_eval.FAILURE))

class RigidBody:

    def __init__(self, prim, dc):
        self.prim = prim
        self._dc = dc
        self.name = prim.GetPrimPath().name
        self.handle = self.get_rigid_body_handle()

    def __repr__(self):
        return self.name

    def get_rigid_body_handle(self):
        object_children = self.prim.GetChildren()
        for child in object_children:
            child_path = child.GetPath().pathString
            body_handle = self._dc.get_rigid_body(child_path)
            if body_handle != 0:
                bin_path = child_path

        object_handle = self._dc.get_rigid_body(bin_path)
        if object_handle != 0: return object_handle

    def get_linear_velocity(self):
        return np.array(self._dc.get_rigid_body_linear_velocity(self.handle))

    def get_angular_velocity(self):
        return np.array(self._dc.get_rigid_body_angular_velocity(self.handle))

    def get_speed(self):
        velocity = self.get_linear_velocity()
        speed = np.linalg.norm(velocity)
        return speed

    def get_pose(self):
        return self._dc.get_rigid_body_pose(self.handle)

    def get_position(self):
        pose = self.get_pose()
        position = np.array(pose.p)
        return position
    
    def get_orientation(self):
        pose = self.get_pose()
        orientation = np.array(pose.r)
        return orientation

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

    def create_franka(self, *args):
        super().create_franka()
        if self.asset_path is None:
            return

        self.objects_usd = [
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
        prim_usd_path = self.objects_usd[random.randint(0, len(self.objects_usd) - 1)]
        prim_env_path = "/scene/objects/object_{}".format(self.current_obj)
        location = Gf.Vec3d(40, 2 * self.current_obj, 10)
        prim = create_prim_from_usd(self._stage, prim_env_path, prim_usd_path, location)
        self.current_obj += 1
        return prim

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

        # register objects
        self.objects = []
        objects_path = '/scene/objects'
        objects_prim = self._stage.GetPrimAtPath(objects_path)
        if objects_prim.IsValid():
            for object_prim in objects_prim.GetChildren():
                self.objects.append(RigidBody(object_prim, self._dc))

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
                    # self._paused = (self._time - self._start_time) < self.timeout_max
                    self._paused = True
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
            else:
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

    def stop_tasks(self, *args):
        if self.pick_and_place is not None:
            if self._editor.is_playing():
                self._reset = True
                self._pending_stop = False
            else:
                self._pending_stop = True

    def pause_tasks(self, *args):
        self._paused = not self._paused
        return self._paused

    # TODO: use gripper.width == 0 as a proxy for _gripper_open == False
    def open_gripper(self):
        if self._gripper_open:
            self.franka_solid.end_effector.gripper.close()
            self._gripper_open = False
        else:
            self.franka_solid.end_effector.gripper.open()
            self._gripper_open = True

    def set_target_angle(self, angle):
        if self.pick_and_place is not None:
            self.pick_and_place.target_angle = angle
