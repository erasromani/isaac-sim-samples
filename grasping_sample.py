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

def create_xyz(init={"X": 30, "Y": 0, "Z": 30}):
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    float_drags = {}
    for axis in all_axis:
        with ui.HStack():            
            with ui.ZStack(width=15):
                ui.Rectangle(
                    width=15,
                    height=20,
                    style={"background_color": colors[axis], "border_radius": 3, "corner_flag": ui.CornerFlag.LEFT},
                )
                ui.Label(axis, name="transform_label", alignment=ui.Alignment.CENTER)
            float_drags[axis] = ui.FloatDrag(name="transform", min=-1000000, max=1000000, step=1, width=100)
            float_drags[axis].model.set_value(init[axis])
    return float_drags

def create_angle(init=0):
    color = 0xFF000000
    with ui.HStack(width=15):
        with ui.ZStack(width=15):
            ui.Rectangle(
                width=15,
                height=20,
                style={"background_color": color, "border_radius": 3, "corner_flag": ui.CornerFlag.LEFT},
            )
            ui.Label("T", name="transform_label", alignment=ui.Alignment.CENTER)
        float_drag = ui.FloatDrag(name="transform", min=-1000000, max=1000000, step=1, width=100)
        float_drag.model.set_value(init)
    return float_drag

def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")  # create an empty Xform at the given path
    envPrim.GetReferences().AddReference(prim_usd_path)  # attach the USD to the given path
    set_translate(envPrim, location)  # set pose

# communication between git and isaac-sim with test branch
class Extension(omni.ext.IExt):
    def on_startup(self):
        """Initialize extension and UI elements
        """
        self._editor = omni.kit.editor.get_editor_interface()
        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()
        self._window = ui.Window(EXTENSION_NAME, width=800, height=400, visible=False)
        self._window.deferred_dock_in("Content")
        self._menu_entry = omni.kit.ui.get_editor_menu().add_item(
            f"Isaac Robotics/Samples/{EXTENSION_NAME}", self._menu_callback
        )
        self._create_ui()

        self._mp = _motion_planning.acquire_motion_planning_interface()
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self._physxIFace = _physx.acquire_physx_interface()

        self._ar = _dynamic_control.INVALID_HANDLE

        self._settings = omni.kit.settings.get_settings_interface()

        self._settings.set("/persistent/physics/updateToUsd", False)
        self._settings.set("/persistent/physics/useFastCache", True)
        self._settings.set("/persistent/physics/numThreads", 8)

        self._termination_criteria = FrameTerminationCriteria(orig_thresh=0.001)

        self._first_step = True
        self._robot = None

        ## unit conversions: RMP is in meters, kit is by default in cm
        self._meters_per_unit = UsdGeom.GetStageMetersPerUnit(self._stage)
        self._units_per_meter = 1.0 / UsdGeom.GetStageMetersPerUnit(self._stage)

    def _menu_callback(self, a, b):
        self._window.visible = not self._window.visible
        if self._window.visible:
            self._sub_stage_event = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event
            )
        else:
            self._sub_stage_event = None

    def _create_ui(self):
        with self._window.frame:
            with omni.ui.VStack():
                with ui.HStack(height=5):
                    ui.Spacer(width=5)
                    self._create_robot_btn = ui.Button("Load Robot", width=125)
                    self._create_robot_btn.set_clicked_fn(self._on_environment_setup)
                    self._created = False  # is the robot loaded
                with ui.HStack(height=5):
                    ui.Spacer(width=5)
                    self._target_following_btn = ui.Button("Target Following", width=125)
                    self._target_following_btn.set_clicked_fn(self._on_target_following)
                    self._target_following_btn.enabled = False
                    self._following = False  # is the task running
                    self._target = None
                with ui.HStack(height=5):
                    ui.Spacer(width=5)
                    self._reset_pose_btn = ui.Button("Reset Robot Pose", width=125)
                    self._reset_pose_btn.set_clicked_fn(self._on_reset_pose)
                    self._reset_pose_btn.enabled = False
                    self._reset_pose_btn.set_tooltip("Reset robot to default position")
                with ui.HStack(height=5):
                    ui.Spacer(width=5)
                    self._add_object_btn = ui.Button("Add Object", width=125)
                    self._add_object_btn.set_clicked_fn(self._on_add_object)
                    self._add_object_btn.enabled = False
                    self._add_object_btn.set_tooltip("Drop randomly selected object in scene")
                with ui.HStack(height=5):
                    ui.Spacer(width=5)
                    self._gripper_btn = ui.Button("Toggle Gripper", width=125)
                    self._gripper_btn.set_clicked_fn(self._on_toggle_gripper)
                    self._gripper_btn.enabled = False
                    self._gripper_open = False
                with ui.HStack(height=5):
                    ui.Spacer(width=9)
                    self._goal_label = ui.Label("Set Grasp Center", width=100)
                    self._goal_label.set_tooltip("Set target grasp center specified as (X, Y, Z)")
                    self.default_goal_coord = {"X": 30, "Y": 0, "Z": 30}
                    self.goal_coord = create_xyz(init=self.default_goal_coord)
                    for axis in self.goal_coord:
                        self.goal_coord[axis].model.add_value_changed_fn(self._on_update_goal_coord)
                with ui.HStack(height=5):
                    ui.Spacer(width=9)
                    self._angle_label = ui.Label("Set Grasp Angle", width=100)
                    self._angle_label.set_tooltip("Set target grasp angle specified in degrees")
                    self.default_goal_angle = 0
                    self.goal_angle = create_angle(init=self.default_goal_angle)
                with ui.HStack(height=5):
                    ui.Spacer(width=5)
                    self._reset_btn = ui.Button("Reset Scene", width=125)
                    self._reset_btn.set_clicked_fn(self._on_reset)
                    self._reset_btn.enabled = False
                    self._reset_btn.set_tooltip("Reset robot and target to default positions")

    def _on_environment_setup(self):
        task = asyncio.ensure_future(omni.kit.asyncapi.new_stage())
        asyncio.ensure_future(self._on_create_robot(task))

    async def _on_create_robot(self, task):
        """ load robot from USD
        """
        done, pending = await asyncio.wait({task})
        if task not in done:
            return

        self._stage = self._usd_context.get_stage()
        self._create_robot_btn.enabled = False

        self._editor.stop()

        set_up_z_axis(self._stage)
        add_ground_plane(self._stage, "/groundPlane", "Z", 1000.0, Gf.Vec3f(0.0), Gf.Vec3f(1.0))
        setup_physics(self._stage)

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        asset_path = nucleus_server + "/Isaac"
        robot_usd = asset_path + "/Robots/Franka/franka.usd"
        robot_path = "/scene/robot"
        create_prim_from_usd(self._stage, robot_path, robot_usd, Gf.Vec3d(0, 0, 0))

        self.objects = [
            asset_path + "/Props/Flip_Stack/large_corner_bracket_physics.usd",
            asset_path + "/Props/Flip_Stack/screw_95_physics.usd",
            asset_path + "/Props/Flip_Stack/screw_99_physics.usd",
            asset_path + "/Props/Flip_Stack/small_corner_bracket_physics.usd",
            asset_path + "/Props/Flip_Stack/t_connector_physics.usd",
        ]
        self.current_obj = 0

        self._physxIFace.release_physics_objects()
        self._physxIFace.force_load_physics_from_usd()

        self._editor_event_subscription = self._editor.subscribe_to_update_events(self._on_editor_step)
        self._physxIFace.release_physics_objects()
        self._physxIFace.force_load_physics_from_usd()
        self._reset_btn.enabled = True

        self._editor.set_camera_position("/OmniverseKit_Persp", 142, -127, 56, True)
        self._editor.set_camera_target("/OmniverseKit_Persp", -180, 234, -27, True)

        light_prim = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/defaultLight"))
        light_prim.CreateIntensityAttr(500)

        self._first_step = True
        self._following = False
        self._robot = None
        self._created = True

    def _register_assets(self):
        ## register world with RMP
        self._world =  World(self._dc, self._mp)

        ## register robot with RMP
        robot_path = "/scene/robot"
        self._robot = Franka(
            self._stage, self._stage.GetPrimAtPath(robot_path), self._dc, self._mp, self._world, default_config
        )

    def _on_update_goal_coord(self, *args):
        goal_x = self.goal_coord["X"].model.get_value_as_float()
        goal_y = self.goal_coord["Y"].model.get_value_as_float()
        goal_z = self.goal_coord["Z"].model.get_value_as_float()
        self._target_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(goal_x, goal_y, goal_z))        

    def _on_target_following(self):
        ## create target
        target_path = "/scene/target"
        if self._stage.GetPrimAtPath(target_path):
            self._following = True
            return

        goal_x = self.goal_coord["X"].model.get_value_as_float()
        goal_y = self.goal_coord["Y"].model.get_value_as_float()
        goal_z = self.goal_coord["Z"].model.get_value_as_float()

        target_geom = UsdGeom.Sphere.Define(self._stage, target_path)
        offset = Gf.Vec3f(goal_x, goal_y, goal_z)  ## these are in cm
        colors = Gf.Vec3f(1.0, 0, 0)
        target_size = 4
        target_geom.CreateRadiusAttr(target_size)
        target_geom.AddTranslateOp().Set(offset)
        target_geom.CreateDisplayColorAttr().Set([colors])
        self._target_prim = self._stage.GetPrimAtPath(target_path)

        ## start following it
        self._following = True

    def _on_toggle_gripper(self):
        if self._gripper_open:
            print("closing gripper")
            self._robot.end_effector.gripper.close()
            self._gripper_open = False
        else:
            print("opening gripper")
            self._robot.end_effector.gripper.open()
            self._gripper_open = True

    def _on_reset_pose(self):
        self._following = False

        # put robot (an articulated prim) in a specific joint configuration
        reset_config = np.array([0.00, -1.3, 0.00, -2.57, 0.00, 2.20, 0.75])
        self._robot.send_config(reset_config)
        self._robot.end_effector.go_local(use_default_config=True, wait_for_target=False)

    def _on_add_object(self):
        prim_usd_path = self.objects[random.randint(0, len(self.objects) - 1)]
        prim_env_path = "/scene/objects/object_{}".format(self.current_obj)
        location = Gf.Vec3d(30, 1.2 * self.current_obj, 10)
        create_prim_from_usd(self._stage, prim_env_path, prim_usd_path, location)
        self.current_obj += 1

    def _on_editor_step(self, step):
        """This function is called every timestep in the editor
        
        Arguments:
            step (float): elapsed time between steps
        """
        self._on_update_ui()
        if self._created and self._editor.is_playing():
            if self._first_step:
                self._register_assets()
                self._first_step = False
            if self._following:
                target_pos = self._target_prim.GetAttribute("xformOp:translate").Get()
                self._target = {"orig": np.array([target_pos[0], target_pos[1], target_pos[2]]) * self._meters_per_unit}
                self._robot.end_effector.go_local(target=self._target, use_default_config=True, wait_for_target=True)
            # update RMP's world and robot states to sync with Kit
            self._world.update()
            self._robot.update()

    def _on_reset(self):
        self._following = False

        # put robot (an articulated prim) in a specific joint configuration
        reset_config = np.array([0.00, -1.3, 0.00, -2.57, 0.00, 2.20, 0.75])
        self._robot.send_config(reset_config)
        self._robot.end_effector.go_local(use_default_config=True, wait_for_target=False)
        self._robot.end_effector.gripper.close()
        self._gripper_open = False

        # put target back (a visual prim) in position
        if self._target:
            for axis in self.goal_coord:
                self.goal_coord[axis].model.set_value(self.default_goal_coord[axis])           
            self._target_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(self.default_goal_coord["X"], self.default_goal_coord["Y"], self.default_goal_coord["Z"]))

        self._robot = None
        self._first_step = True

    def _stop_tasks(self):
        self._following = False
        self._robot = None
        self._created = False
        gc.collect()

    def _on_stage_event(self, event):
        """This function is called when stage events occur.
        Enables UI elements when stage is opened.
        Prevents tasks from being started until all assets are loaded
        
        Arguments:
            event (int): event type
        """
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._create_robot_btn.enabled = True
            self._target_following_btn.enabled = False
            self._reset_pose_btn.enabled = False
            self._add_object_btn.enabled = False
            self._gripper_btn.enabled = False
            self._reset_btn.enabled = False
            self._editor.stop()
            self._stop_tasks()

    def _on_update_ui(self):
        """Callback that updates UI elements every frame
        """
        if self._created:
            self._create_robot_btn.enabled = True
            self._target_following_btn.enabled = False
            self._add_object_btn.enabled = False
            self._gripper_btn.enabled = False
            self._reset_pose_btn.enabled = False
            self._reset_btn.enabled = False
            if self._editor.is_playing():
                self._reset_pose_btn.enabled = True
                self._target_following_btn.enabled = True
                self._target_following_btn.text = "Follow Target"
                self._add_object_btn.enabled = True
                self._gripper_btn.enabled = True
                self._reset_btn.enabled = True
                if self._gripper_open:
                    self._gripper_btn.text = "Press to Close Gripper"
                else:
                    self._gripper_btn.text = "Press to Open Gripper"
            else:
                self._target_following_btn.enabled = False
                self._target_following_btn.text = "Press Play To Enable"

        else:
            self._create_robot_btn.enabled = True
            self._target_following_btn.text = "Press Create To Enable"

    def on_shutdown(self):
        """Cleanup objects on extension shutdown
        """
        self._editor.stop()
        self._stop_tasks()
        self._editor_event_subscription = None
        self._window = None
        gc.collect()

    def has_arrived(self):
        """if multiple targets are sent, the later one will overwrite the earlier one. 
            Use this function to check for arrived condition to be met before going to the next target.
        """
        return self._termination_criteria(self._target, self._robot.end_effector.status.current_frame)
