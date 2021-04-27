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
    
    def load_single_object(self, drop=False, max_steps=1000):
        self.add_and_register_object()
        if drop:
            # start simulation
            kit.play()
            step = 0
            target_object = self.objects[-1]
            while step < max_steps or self._kit.is_loading():
                self.step(step)
                self._kit.update(1 / 60.0)
                object_speed = target_object.get_speed()
                if object_speed == 0: break
                step +=1
            # Stop physics simulation
            kit.stop()

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

_editor.set_camera_position("/OmniverseKit_Persp", 142, -127, 56, True)
_editor.set_camera_target("/OmniverseKit_Persp", -180, 234, -27, True)


kit.play()

_scenario.register_assets()
_scenario.load_single_object(drop=False)
kit.stop()

# # start simulation
# kit.play()
   
# _scenario.register_assets()
# screenshot(sd_helper)
# object_x_positions = []
# object_y_positions = []
# object_z_positions = []
# object_velocities = []

# step = 0
# target_object = _scenario.objects[0]
# while step < 1000 or kit.is_loading():
#     _scenario.step(step)
#     kit.update(1 / 60.0)
#     object_velocity = target_object.get_speed()
#     object_position = target_object.get_position()
#     object_x_positions.append(object_position[0])
#     object_y_positions.append(object_position[1])
#     object_z_positions.append(object_position[2])
#     object_velocities.append(object_velocity)
#     if object_velocity == 0: break
#     step +=1
# screenshot(sd_helper, suffix=1)

# fig, axes = plt.subplots(2, 2)
# axes.flatten()[0].plot(object_velocities)
# axes.flatten()[0].set_ylabel('velocity')
# axes.flatten()[1].plot(object_x_positions)
# axes.flatten()[1].set_ylabel('x position')
# axes.flatten()[2].plot(object_y_positions)
# axes.flatten()[2].set_ylabel('y position')
# axes.flatten()[3].plot(object_z_positions)
# axes.flatten()[3].set_ylabel('z position')
# plt.savefig('images/velocity.png')

# # Stop physics simulation
# kit.stop()