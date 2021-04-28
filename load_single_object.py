# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# import carb.input
# import omni.kit.commands
# import omni.kit.editor
# import omni.ext
# import omni.kit.ui
# import omni.kit.settings

import carb

from pxr import Gf, UsdGeom
from omni.isaac.motion_planning import _motion_planning
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx import _physx

from grasping_scenarios.scenario import Scenario
from grasping_scenarios.grasp_object import GraspObject, RigidBody
import asyncio

import sys
import matplotlib.pyplot as plt

sys.path.append('/home/robot-lab/isaac-sim/_build/linux-x86_64/release/exts')

import os
import omni
import random
import numpy as np
from pxr import UsdGeom, Semantics
from omni.isaac.synthetic_utils import OmniKitHelper
from omni.isaac.synthetic_utils import SyntheticDataHelper
from utils.visualize import screenshot


class GraspSimulator(GraspObject):
    """ Defines an obstacle avoidance scenario

    Scenarios define the life cycle within kit and handle init, startup, shutdown etc.
    """

    def __init__(self, kit, dc, mp):
        super().__init__(kit.editor, dc, mp)
        self._kit = kit
        self.frame = 0
    
    def load_single_object(self, drop=False, max_steps=2000):
        self.add_and_register_object()
        if drop:
            # start simulation
            if self._kit.editor.is_playing(): previously_playing = True
            else:                             previously_playing = False

            if not previously_playing: self.play()
            step = 0
            target_object = self.objects[-1]
            while step < max_steps or self._kit.is_loading():
                self.step(step)
                self._kit.update(1 / 60.0)
                object_speed = target_object.get_speed()
                if object_speed == 0: break
                step +=1

            # Stop physics simulation
            if not previously_playing: self.stop()

    def execute_grasp(self, position, angle):
        self.set_target_angle(angle)
        self.perform_tasks()
        # start simulation
        if self._kit.editor.is_playing(): previously_playing = True
        else:                             previously_playing = False

        if self.pick_and_place is not None:

            while True:
                self.step(0)
                self._kit.update(1 / 60.0)
                self.frame += 1
                if self.pick_and_place.evaluation is not None:
                    break
                if self.frame % 10 == 0: screenshot(sd_helper, suffix=self.frame)
        evaluation = self.pick_and_place.evaluation
        self.stop_tasks()
        self.step(0)
        self._kit.update(1 / 60.0)

        # Stop physics simulation
        if not previously_playing: self.stop()

        return evaluation

    def add_and_register_object(self):
        prim = self.create_new_objects()
        self._kit.update()
        if not hasattr(self, 'objects'):
            self.objects = []
        self.objects.append(RigidBody(prim, self._dc))

    def play(self):
        self._kit.play()
    
    def stop(self):
        self._kit.stop()

kit = OmniKitHelper(
    {"renderer": "RayTracedLighting", "experience": f'{os.environ["EXP_PATH"]}/isaac-sim-python.json'}
)
sd_helper = SyntheticDataHelper()

_stage = kit.get_stage()
_editor = kit.editor

_mp = _motion_planning.acquire_motion_planning_interface()
_dc = _dynamic_control.acquire_dynamic_control_interface()
_physxIFace = _physx.acquire_physx_interface()
_scenario = GraspSimulator(kit, _dc, _mp)

_scenario.create_franka()

# _editor.set_camera_position("/OmniverseKit_Persp", 142, -127, 56, True)
# _editor.set_camera_target("/OmniverseKit_Persp", -180, 234, -27, True)

camera_rig = UsdGeom.Xformable(kit.create_prim("/scene/robot/panda_hand/CameraRig", "Xform"))
camera = kit.create_prim("/scene/robot/panda_hand/Camera", "Camera", translation=(0.0, 0.0, 11), rotation=(180, 0, 0))
vpi = omni.kit.viewport.get_viewport_interface()
vpi.get_viewport_window().set_active_camera(str(camera.GetPath()))

# start simulation
kit.play()

_scenario.register_assets()
# _scenario.load_single_object(drop=True)

while kit.is_loading():
    kit.update(1 / 60.0)


evaluation = _scenario.execute_grasp(0, 0)

print(evaluation)

# evaluation = _scenario.execute_grasp(0, 90)

# print(evaluation)

# evaluation = _scenario.execute_grasp(0, 45)

# print(evaluation)

# evaluation = _scenario.execute_grasp(0, 10)

# print(evaluation)

# evaluation = _scenario.execute_grasp(0, -20)

# print(evaluation)

# evaluation = _scenario.execute_grasp(0, 100)

# print(evaluation)

# Stop physics simulation
kit.stop()
