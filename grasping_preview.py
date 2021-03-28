# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb.input
from pxr import Usd, UsdGeom
import omni.kit.commands
import omni.kit.editor
import omni.ext
import omni.appwindow
import omni.kit.ui
import omni.kit.settings
import asyncio

from omni.isaac.motion_planning import _motion_planning
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx import _physx

from .franka_scenarios.scenario import Scenario
from .franka_scenarios.ghost_scenario import GhostScenario
from .franka_scenarios.simple_stack import SimpleStack
from .franka_scenarios.multiple_obstacle import MultipleObstacle

EXTENSION_NAME = "Leonardo Preview"


class Extension(omni.ext.IExt):
    def on_startup(self):
        """Initialize extension and UI elements
        """
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

        self._mp = _motion_planning.acquire_motion_planning_interface()
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._physxIFace = _physx.acquire_physx_interface()

        self._selected_scenario = self._window.layout.add_child(omni.kit.ui.ComboBox())
        self._selected_scenario.add_item("Ghost Robots")
        self._selected_scenario.add_item("Simple Stack")
        self._selected_scenario.add_item("Multiple Obstacles")
        self._selected_scenario.selected_index = 0

        self._create_franka_btn = self._window.layout.add_child(omni.kit.ui.Button("Create Scenario"))
        self._create_franka_btn.set_clicked_fn(self._on_environment_setup)

        self._perform_task_btn = self._window.layout.add_child(omni.kit.ui.Button("Perform Task"))
        self._perform_task_btn.set_clicked_fn(self._on_perform_task)
        self._perform_task_btn.enabled = False

        self._stop_task_btn = self._window.layout.add_child(omni.kit.ui.Button("Stop/Reset Task"))
        self._stop_task_btn.set_clicked_fn(self._on_stop_tasks)
        self._stop_task_btn.enabled = False

        self._toggle_obstacle_btn = self._window.layout.add_child(omni.kit.ui.Button("Toggle Obstacle"))
        self._toggle_obstacle_btn.set_clicked_fn(self._on_toggle_obstacle)
        self._toggle_obstacle_btn.enabled = False

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

    def _on_environment_setup(self, widget):
        # wait for new stage before creating franka
        task = asyncio.ensure_future(omni.kit.asyncapi.new_stage())
        asyncio.ensure_future(self._on_create_franka(task))

    async def _on_create_franka(self, task):
        """Load any assets required by the scenario and create objects
        """
        done, pending = await asyncio.wait({task})
        if task not in done:
            return

        self._stage = self._usd_context.get_stage()

        if self._selected_scenario.selected_index == 0:
            self._scenario = GhostScenario(self._editor, self._dc, self._mp)
        elif self._selected_scenario.selected_index == 1:
            self._scenario = SimpleStack(self._editor, self._dc, self._mp)
        elif self._selected_scenario.selected_index == 2:
            self._scenario = MultipleObstacle(self._editor, self._dc, self._mp)

        self._create_franka_btn.enabled = False
        self._selected_scenario.enabled = False

        self._editor.stop()
        self._physxIFace.release_physics_objects()

        self._settings.set("/rtx/reflections/halfRes", True)
        self._settings.set("/rtx/shadows/denoiser/quarterRes", True)
        self._settings.set("/rtx/translucency/reflectionCutoff", 0.1)

        self._scenario.create_franka()

        self._physxIFace.release_physics_objects()
        self._physxIFace.force_load_physics_from_usd()

        self._scenario.register_assets()

        self._editor_event_subscription = self._editor.subscribe_to_update_events(self._on_editor_step)
        self._physxIFace.release_physics_objects()
        self._physxIFace.force_load_physics_from_usd()
        self._stop_task_btn.enabled = True
        self._toggle_obstacle_btn.enabled = True

        self._editor.set_camera_position("/OmniverseKit_Persp", 142, -127, 56, True)
        self._editor.set_camera_target("/OmniverseKit_Persp", -180, 234, -27, True)

        light_prim = self._stage.GetPrimAtPath("/World/defaultLight")
        if light_prim:
            light_prim.SetActive(False)

    def _on_stop_tasks(self, *args):
        """Stop all tasks being performed by the scenario
        """
        self._scenario.stop_tasks()

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events
        press 1 to perform tasks
        press 2 to stop tasks
        press 3 to toggle obstacle

        Args:
            event (int): keyboard event type
        """

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.KEY_1:
                self._on_perform_task()
            if event.input == carb.input.KeyboardInput.KEY_2:
                self._on_stop_tasks()
            if event.input == carb.input.KeyboardInput.KEY_3:
                self._on_toggle_obstacle()
        return True

    def _on_editor_step(self, step):
        """This function is called every timestep in the editor

        Arguments:
            step (float): elapsed time between steps
        """
        self._scenario.step(step)

    def _on_stage_event(self, event):
        """This function is called when stage events occur.
        Enables UI elements when stage is opened.
        Prevents tasks from being started until all assets are loaded

        Arguments:
            event (int): event type
        """
        self.stage = self._usd_context.get_stage()
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._create_franka_btn.enabled = True
            self._selected_scenario.enabled = True
            self._perform_task_btn.enabled = False
            self._stop_task_btn.enabled = False
            self._toggle_obstacle_btn.enabled = False
            self._editor.stop()
            self._on_stop_tasks()
            self._scenario = Scenario(self._editor, self._dc, self._mp)

    def _on_toggle_obstacle(self, *args):
        """Toggle obstacle visibility
        """
        for obstacle in self._scenario._obstacles:
            imageable = UsdGeom.Imageable(self._stage.GetPrimAtPath(obstacle.asset_path))
            visibility = imageable.ComputeVisibility(Usd.TimeCode.Default())
            if visibility == UsdGeom.Tokens.invisible:
                imageable.MakeVisible()
                obstacle.unsuppress()
            else:
                imageable.MakeInvisible()
                obstacle.suppress()

    def _on_perform_task(self, *args):
        """Perform all tasks in the scenario
        """
        self._scenario.perform_tasks()

    def _on_update_ui(self, widget):
        """Callback that updates UI elements every frame
        """
        if self._scenario.is_created():
            self._create_franka_btn.enabled = False
            self._perform_task_btn.enabled = False
            self._stop_task_btn.enabled = False
            if self._editor.is_playing():
                self._perform_task_btn.enabled = True
                self._perform_task_btn.text = "Perform Task"
                if self._scenario._running is True:
                    self._perform_task_btn.enabled = False
                    self._stop_task_btn.enabled = True
                else:
                    self._perform_task_btn.enabled = True
                    self._stop_task_btn.enabled = False
            else:
                self._perform_task_btn.enabled = False
                self._perform_task_btn.text = "Press Play To Enable"
                self._scenario._running = False
        else:
            self._create_franka_btn.enabled = True
            self._perform_task_btn.enabled = False
            self._perform_task_btn.text = "Press Create To Enable"
            self._stop_task_btn.enabled = False
            self._toggle_obstacle_btn.enabled = False

    def on_shutdown(self):
        """Cleanup objects on extension shutdown
        """
        self._editor.stop()
        self._on_stop_tasks()
        self._scenario = None
        self._editor_event_subscription = None
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
        self._window.set_update_fn(None)
