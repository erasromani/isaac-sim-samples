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
from grasping_scenarios.grasp_object import GraspObject
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


CAMERA_DISTANCE = 300



kit = OmniKitHelper(
    {"renderer": "RayTracedLighting", "experience": f'{os.environ["EXP_PATH"]}/isaac-sim-python.json'}
)
sd_helper = SyntheticDataHelper()

_stage = kit.get_stage()
_editor = kit.editor

_mp = _motion_planning.acquire_motion_planning_interface()
_dc = _dynamic_control.acquire_dynamic_control_interface()
_physxIFace = _physx.acquire_physx_interface()
_scenario = GraspObject(_editor, _dc, _mp)

_scenario.create_franka()

camera_rig = UsdGeom.Xformable(kit.create_prim("/World/CameraRig", "Xform"))
camera = kit.create_prim("/World/CameraRig/Camera", "Camera", translation=(0.0, 0.0, CAMERA_DISTANCE))
# camera_rig = UsdGeom.Xformable(kit.create_prim("/scene/robot/panda_hand/CameraRig", "Xform"))
# camera = kit.create_prim("/scene/robot/panda_hand/Camera", "Camera", translation=(0.0, 0.0, 11), rotation=(180, 0, 0))
# vpi = omni.kit.viewport.get_viewport_interface()
# vpi.get_viewport_window().set_active_camera(str(camera.GetPath()))

_editor.set_camera_position("/OmniverseKit_Persp", 142, -127, 56, True)
_editor.set_camera_target("/OmniverseKit_Persp", -180, 234, -27, True)

kit.update()

# start simulation
kit.play()

while kit.is_loading():
    kit.update(1 / 60.0)

step = 0

_scenario.perform_tasks()

# TODO: Update pick_and_place method to allow grasp position and orientation input (i.e. angle and location)
# TODO: change the pick_and_place method to one_shot not cyclic
# TODO: remove all execution of time.sleep in the code
# TODO: turn this file into a class structure that encompasses the kit object and the _scenario object. There should also be a function that clears the scene or at least the unnecessary objects
while step < 1000:
    if step == 0:
        _scenario.register_assets()
    _scenario.step(step)
    kit.update(1 / 60.0)

    if step % 100 == 0:
        gt = sd_helper.get_groundtruth(
            [
                "rgb",
                # "camera",
                # "depth",
                # "boundingBox2DTight",
                # "boundingBox2DLoose",
                # "instanceSegmentation",
                # "semanticSegmentation",
                # "boundingBox3D",
            ]
        )

        image = gt["rgb"][..., :3]
        if _scenario.pick_and_place.evaluation:
            break
        plt.imshow(image)
        plt.savefig(f'images/test_{step:05}.png')
    step +=1

carb.log_warn(str(_scenario.pick_and_place.evaluation))

# Stop physics simulation
kit.stop()