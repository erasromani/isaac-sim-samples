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
from omni.isaac.samples.scripts.utils.franka import Franka, default_config

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


statedic = {0: "orig", 1: "axis_x", 2: "axis_y", 3: "axis_z"}


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
        self._gripper_open = True

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
        self.default_position.p = [0.3, 0.0, 0.3]
        self.default_position.r = [0.0, 1.0, 0.0, 0.0]
        p = self.default_position.p
        r = self.default_position.r
        set_translate(GoalPrim, Gf.Vec3d(p.x * 100, p.y * 100, p.z * 100))
        set_rotate(GoalPrim, Gf.Matrix3d(Gf.Quatd(r.w, r.x, r.y, r.z)))

        # Setup physics simulation
        add_ground_plane(self._stage, "/groundPlane", "Z", 1000.0, Gf.Vec3f(0.0), Gf.Vec3f(1.0))
        setup_physics(self._stage)

    def add_bin(self, *args):
        self.create_new_objects(args)

    def create_new_objects(self, *args):
        prim_usd_path = self.objects[random.randint(0, len(self.objects) - 1)]
        prim_env_path = "/scene/objects/object_{}".format(self.current_obj)
        location = Gf.Vec3d(30, 2 * self.current_obj, 10)
        create_prim_from_usd(self._stage, prim_env_path, prim_usd_path, location)
        self.current_obj += 1

    def register_assets(self, *args):

        ## register world with RMP
        self.world =  World(self._dc, self._mp)

        ## register robot with RMP
        robot_path = "/scene/robot"
        self.franka_solid = Franka(
            self._stage, self._stage.GetPrimAtPath(robot_path), self._dc, self._mp, self.world, default_config
        )

        # register objects

        # TODO: registier stage machine 
        # self.pick_and_place = PickAndPlaceStateMachine(
        #     self._stage,
        #     self.ur10_solid,
        #     self._stage.GetPrimAtPath(self.env_path + "/ur10/ee_link"),
        #     self.bin_path,
        #     self.default_position,
        # )

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
                # self.pick_and_place.step(self._time, self._start, self._reset)
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
                    axis_x=axis_x,
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
