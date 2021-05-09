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

# import carb

# from pxr import Gf, UsdGeom
from omni.isaac.motion_planning import _motion_planning
from omni.isaac.dynamic_control import _dynamic_control
# from omni.physx import _physx

# from grasping_scenarios.scenario import Scenario
from grasping_scenarios.grasp_object import GraspObject, RigidBody
# import asyncio

# import sys
# import matplotlib.pyplot as plt

# sys.path.append('/home/robot-lab/isaac-sim/_build/linux-x86_64/release/exts')

import os
import numpy as np
import tempfile
# import time
# import omni
# import random
# import numpy as np
# from pxr import UsdGeom, Semantics
from omni.isaac.synthetic_utils import OmniKitHelper
from omni.isaac.synthetic_utils import SyntheticDataHelper
from utils.visualize import screenshot, img2vid

default_camera_pose = {
    'position': (142, -127, 56),
    'target': (-180, 234, -27)
    }

class GraspSimulator(GraspObject):
    """ Defines a grasping simulation scenario

    Scenarios define planar grasp execution in a scene of a Panda arm and various rigid objects
    """

    def __init__(self, kit, dc, mp, dt=1/60.0, record=False, record_interval=10):
        super().__init__(kit, dc, mp)
        self.frame = 0
        self.dt = dt
        self.record = record
        self.record_interval = record_interval
        self.tmp_dir = tempfile.mkdtemp()
        self.sd_helper = SyntheticDataHelper()
        
        # create initial scene
        self.create_franka()

        # set camera pose
        self.set_camera_pose(default_camera_pose['position'], default_camera_pose['target'])

    def load_object(self, drop=False, max_steps=2000):
        """
        """
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
                self.update()
                object_speed = target_object.get_speed()
                if object_speed == 0: break
                step +=1

            # Stop physics simulation
            if not previously_playing: self.stop()

    def wait_for_drop(self, max_steps=2000):
        # start simulation
        if self._kit.editor.is_playing(): previously_playing = True
        else:                             previously_playing = False

        if not previously_playing: self.play()
        step = 0
        while step < max_steps or self._kit.is_loading():
            self.step(step)
            self.update()
            objects_speed = np.array([o.get_speed() for o in self.objects])
            if np.all(objects_speed == 0): break
            step +=1

        # Stop physics simulation
        if not previously_playing: self.stop()

    def execute_grasp(self, position, angle):
        """
        """
        self.set_target_angle(angle)
        self.set_target_position(position)
        self.perform_tasks()
        # start simulation
        if self._kit.editor.is_playing(): previously_playing = True
        else:                             previously_playing = False

        if self.pick_and_place is not None:

            while True:
                self.step(0)
                self.update()
                if self.pick_and_place.evaluation is not None:
                    break
        evaluation = self.pick_and_place.evaluation
        self.stop_tasks()
        self.step(0)
        self.update()

        # Stop physics simulation
        if not previously_playing: self.stop()

        return evaluation

    def add_and_register_object(self):
        prim = self.create_new_objects()
        self._kit.update()
        if not hasattr(self, 'objects'):
            self.objects = []
        self.objects.append(RigidBody(prim, self._dc))

    def wait_for_loading(self):
        while self.is_loading():
            self.update()

    def play(self):
        """
        """
        self._kit.play()
        if not hasattr(self, 'world') or not hasattr(self, 'franka_solid') or not hasattr(self, 'bin_solid') or not hasattr(self, 'pick_and_place'):
            self.register_scene()
    
    def stop(self):
        """
        """
        self._kit.stop()

    def update(self):
        """
        """
        if self.record and self.sd_helper is not None and self.frame % self.record_interval == 0:     

            screenshot(self.sd_helper, suffix=self.frame, directory=self.tmp_dir)
        
        self._kit.update(self.dt)
        self.frame += 1

    def is_loading(self):
        """
        """
        return self._kit.is_loading()

    def set_camera_pose(self, position, target):
        """
        """
        self._editor.set_camera_position("/OmniverseKit_Persp", *position, True)
        self._editor.set_camera_target("/OmniverseKit_Persp", *target, True)

    def save_video(self, path):
        """
        """
        img2vid(os.path.join(self.tmp_dir, '*.png'), path)   

kit = OmniKitHelper(
    {"renderer": "RayTracedLighting", "experience": f'{os.environ["EXP_PATH"]}/isaac-sim-python.json'}
)
_mp = _motion_planning.acquire_motion_planning_interface()
_dc = _dynamic_control.acquire_dynamic_control_interface()

sim = GraspSimulator(kit, _dc, _mp, record=True)

# make this part of GraspSimulator init
# sim.create_franka()
sim.add_object_path("Isaac/Props/Flip_Stack/large_corner_bracket_physics.usd", from_server=True)
sim.add_object_path("Isaac/Props/Flip_Stack/screw_95_physics.usd", from_server=True)
sim.add_object_path("Isaac/Props/Flip_Stack/screw_99_physics.usd", from_server=True)
sim.add_object_path("Isaac/Props/Flip_Stack/small_corner_bracket_physics.usd", from_server=True)
sim.add_object_path("Isaac/Props/Flip_Stack/t_connector_physics.usd", from_server=True)
# sim.add_object()

# start simulation
sim.play()

# sim.register_scene()
sim.load_object(drop=False) # option to set object spawn pose
sim.load_object(drop=False) # option to set object spawn pose
sim.load_object(drop=False) # option to set object spawn pose
sim.load_object(drop=False) # option to set object spawn pose
sim.load_object(drop=False) # option to set object spawn pose

sim.wait_for_drop()
sim.wait_for_loading()

evaluation = sim.execute_grasp([40, 0, 5], 0)

print(evaluation)

# Stop physics simulation
sim.stop()
sim.save_video('test7.mp4')