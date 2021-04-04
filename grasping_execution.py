# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb.input
import omni.kit.commands
import omni.kit.editor
import omni.ext
import omni.kit.ui
import omni.kit.settings

from pxr import Gf, UsdGeom
from omni.isaac.motion_planning import _motion_planning
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx import _physx

from .grasping_scenarios.scenario import Scenario
from .grasping_scenarios.grasp_object import GraspObject
import asyncio

EXTENSION_NAME = "Grasping Execution"


class Extension(omni.ext.IExt):
    def on_startup(self):
        self._editor = omni.kit.editor.get_editor_interface()
        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()
        self._window = omni.kit.ui.Window(
            EXTENSION_NAME,
            300,
            200,
            menu_path="Isaac Robotics/Samples/" + EXTENSION_NAME,
            open=False,
            dock=omni.kit.ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.set_update_fn(self._on_update_ui)

        self._first_step = True
        self._is_playing = False

        self._mp = _motion_planning.acquire_motion_planning_interface()
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._physxIFace = _physx.acquire_physx_interface()

        self._create_franka_btn = self._window.layout.add_child(omni.kit.ui.Button("Create Scenario"))
        self._create_franka_btn.set_clicked_fn(self._on_environment_setup)

        self._perform_task_btn = self._window.layout.add_child(omni.kit.ui.Button("Perform Task"))
        self._perform_task_btn.set_clicked_fn(self._on_perform_task)
        self._perform_task_btn.enabled = False

        self._stop_task_btn = self._window.layout.add_child(omni.kit.ui.Button("Reset Task"))
        self._stop_task_btn.set_clicked_fn(self._on_stop_tasks)
        self._stop_task_btn.enabled = False

        self._pause_task_btn = self._window.layout.add_child(omni.kit.ui.Button("Pause Task"))
        self._pause_task_btn.set_clicked_fn(self._on_pause_tasks)
        self._pause_task_btn.enabled = False
        
        self._add_object_btn = self._window.layout.add_child(omni.kit.ui.Button("Add Object"))
        #self._add_object_btn.set_clicked_fn(self._on_add_object)
        self._add_object_btn.enabled = False
        self._add_object_btn.tooltip = omni.kit.ui.Label("Drop randomly selected object in scene")

        self._open_gripper_btn = self._window.layout.add_child(omni.kit.ui.Button("Toggle Gripper"))
        #self._open_gripper_btn.set_clicked_fn(self._on_open_gripper)
        self._open_gripper_btn.enabled = False

        self._reset_btn = self._window.layout.add_child(omni.kit.ui.Button("Reset Scene"))
        #self._reset_btn.set_clicked_fn(self._on_reset)
        self._reset_btn.enabled = False
        self._reset_btn.tooltip = omni.kit.ui.Label("Reset robot and target to default positions")

        self._settings = omni.kit.settings.get_settings_interface()

        self._settings.set("/persistent/physics/updateToUsd", False)
        self._settings.set("/persistent/physics/useFastCache", True)
        self._settings.set("/persistent/physics/numThreads", 8)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._sub_stage_event = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event
        )
        self._scenario = Scenario(self._editor, self._dc, self._mp)

        ## unit conversions: RMP is in meters, kit is by default in cm
        self._meters_per_unit = UsdGeom.GetStageMetersPerUnit(self._stage)
        self._units_per_meter = 1.0 / UsdGeom.GetStageMetersPerUnit(self._stage)

    def _on_clear_scenario(self, widget):
        # wait for new stage before creating franka
        asyncio.ensure_future(omni.kit.asyncapi.new_stage())
        self._create_franka_btn.text = "Create Scenario"
        self._create_franka_btn.set_clicked_fn(self._on_environment_setup)

    def _on_environment_setup(self, widget):
        # wait for new stage before creating franka
        task = asyncio.ensure_future(omni.kit.asyncapi.new_stage())
        asyncio.ensure_future(self._on_create_robot(task))
        self._create_franka_btn.text = "Clear Scenario"
        self._create_franka_btn.set_clicked_fn(self._on_clear_scenario)

    async def _on_create_robot(self, task):

        done, pending = await asyncio.wait({task})
        if task not in done:
            return

        self._stage = self._usd_context.get_stage()

        self._scenario = GraspObject(self._editor, self._dc, self._mp)
        self._editor.set_camera_position("/OmniverseKit_Persp", 142, -127, 56, True)
        self._editor.set_camera_target("/OmniverseKit_Persp", -180, 234, -27, True)

        self._first_step = True

        self._editor.stop()
        self._physxIFace.release_physics_objects()

        self._settings.set("/rtx/reflections/halfRes", True)
        self._settings.set("/rtx/shadows/denoiser/quarterRes", True)
        self._settings.set("/rtx/translucency/reflectionCutoff", 0.1)

        self._scenario.create_franka()

        self._physxIFace.release_physics_objects()
        self._physxIFace.force_load_physics_from_usd()

        self._editor_event_subscription = self._editor.subscribe_to_update_events(self._on_editor_step)
        self._physxIFace.release_physics_objects()
        self._physxIFace.force_load_physics_from_usd()
        self._reset_btn.enabled = True
        self._stop_task_btn.enabled = True
        self._pause_task_btn.enabled = True
        self._add_object_btn.enabled = True

    def _on_stop_tasks(self, *args):
        self._scenario.stop_tasks()

    def _on_pause_tasks(self, *args):
        self._open_gripper_btn.enabled = self._scenario.pause_tasks()
        self._perform_task_btn.enabled = not self._open_gripper_btn.enabled
        self._stop_task_btn.enabled = True

    def _on_open_gripper(self, *args):
        self._scenario.open_gripper()

    def _on_add_bin(self, *args):
        # self._scenario.add_bin()
        pass

    def _sub_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.KEY_1:
                self._on_perform_task()
            if event.input == carb.input.KeyboardInput.KEY_2:
                self._on_pause_tasks()
            if event.input == carb.input.KeyboardInput.KEY_3:
                self._on_add_bin()
        return True

    def _on_editor_step(self, step):
        if self._editor.is_playing():
            if self._first_step:
                self._scenario.register_assets()
                self._first_step = False
            self._scenario.step(step)

    def _on_stage_event(self, event):
        self._stage = self._usd_context.get_stage()
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._create_franka_btn.enabled = True
            self._perform_task_btn.enabled = False
            self._stop_task_btn.enabled = False
            self._pause_task_btn.enabled = False
            self._open_gripper_btn.enabled = False
            self._add_object_btn.enabled = False
            self._editor.stop()
            self._on_stop_tasks()
            self._scenario = Scenario(self._editor, self._dc, self._mp)

    def _on_perform_task(self, *args):
        self._perform_task_btn.enabled = False
        self._pause_task_btn.enabled = True
        self._stop_task_btn.enabled = True
        self._open_gripper_btn.enabled = False
        self._scenario.perform_tasks()

    def _on_update_ui(self, widget):
        is_stopped = self._editor.is_stopped()
        if is_stopped and self._is_playing:
            self._on_stop_tasks()
        self._is_playing = not is_stopped

        if self._editor.is_playing() or self._scenario.is_created():
            self._perform_task_btn.enabled = self._scenario._paused
            #self._open_gripper_btn.enabled = True
            self._add_object_btn.enabled = True
            self._perform_task_btn.text = "Perform Task"
            if not self._scenario.is_created():
                self._perform_task_btn.enabled = False
                self._perform_task_btn.text = "Press Create To Enable"
        if not self._editor.is_playing():
            self._perform_task_btn.enabled = False
            #self._open_gripper_btn.enabled = False
            self._add_object_btn.enabled = False
            self._perform_task_btn.text = "Press Play To Enable"
            if not self._scenario.is_created():
                self._create_franka_btn.enabled = True
                self._perform_task_btn.text = "Press Create To Enable"

    def on_shutdown(self):
        self._editor.stop()
        self._on_stop_tasks()
        self._scenario = None
        self._editor_event_subscription = None
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
        self._window.set_update_fn(None)
