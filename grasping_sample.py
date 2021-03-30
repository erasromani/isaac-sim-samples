# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
import random
from pxr import Usd, UsdGeom, Gf, PhysicsSchema, PhysxSchema, Sdf, UsdLux
import omni.kit.editor
import omni.ext
import omni.usd
import omni.kit.ui
import omni.ui as ui
import omni.kit.settings
import asyncio

from omni.isaac.motion_planning import _motion_planning
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx import _physx

from omni.physx.scripts.physicsUtils import add_ground_plane
from omni.isaac.samples.scripts.utils.franka import Franka, default_config

from omni.isaac.samples.scripts.utils.world import World
from omni.isaac.samples.scripts.utils.reactive_behavior import FrameTerminationCriteria
from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
from omni.isaac.utils.scripts.scene_utils import set_translate, set_up_z_axis, setup_physics, create_background

import numpy as np
import gc

EXTENSION_NAME = "Grasping Sample"

# communication between git and isaac-sim with test branch
class Extension(omni.ext.IExt):
    def on_startup(self):
        """Initialize extension and UI elements
        """
        self._window = omni.kit.ui.Window(
            EXTENSION_NAME,
            300,
            200,
            menu_path="Isaac Robotics/Samples/" + EXTENSION_NAME,
            open=False,
            dock=omni.kit.ui.DockPreference.LEFT_BOTTOM,
        )


    def on_shutdown(self):
        """Cleanup objects on extension shutdown
        """
        gc.collect()
