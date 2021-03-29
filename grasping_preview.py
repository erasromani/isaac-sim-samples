# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
from pxr import Usd, UsdGeom, Gf, PhysicsSchema, PhysxSchema, Sdf, UsdLux
import omni.kit.editor
import omni.ext
import omni.usd
import omni.kit.ui
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

EXTENSION_NAME = "Grasping Preview"


def create_prim_from_usd(stage, prim_env_path, prim_usd_path, location):
    envPrim = stage.DefinePrim(prim_env_path, "Xform")  # create an empty Xform at the given path
    envPrim.GetReferences().AddReference(prim_usd_path)  # attach the USD to the given path
    set_translate(envPrim, location)  # set pose

# testing
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

        self._create_robot_btn = self._window.layout.add_child(omni.kit.ui.Button("Load Robot"))
        self._create_robot_btn.set_clicked_fn(self._on_environment_setup)
        self._created = False  # is the robot loaded

        self._target_following_btn = self._window.layout.add_child(omni.kit.ui.Button("Target Following"))
        self._target_following_btn.set_clicked_fn(self._on_target_following)
        self._target_following_btn.enabled = False
        self._following = False  # is the task running
        self._target = None

        self._grasp_object_btn = self._window.layout.add_child(omni.kit.ui.Button("Grasp Object"))
        self._grasp_object_btn.set_clicked_fn(self._on_grasp_object)
        self._grasp_object_btn.enabled = False

        self._add_obstacle_btn = self._window.layout.add_child(omni.kit.ui.Button("Add Obstacles"))
        self._add_obstacle_btn.set_clicked_fn(self._on_add_obstacle)
        self._add_obstacle_btn.enabled = False
        self._obstacle_on = True  # is the obstacle active
        self._block_prim = None

        self._toggle_obstacle_btn = self._window.layout.add_child(omni.kit.ui.Button("Toggle Obstacles"))
        self._toggle_obstacle_btn.set_clicked_fn(self._on_toggle_obstacle)
        self._toggle_obstacle_btn.enabled = False

        self._gripper_btn = self._window.layout.add_child(omni.kit.ui.Button("Toggle Gripper"))
        self._gripper_btn.set_clicked_fn(self._on_toggle_gripper)
        self._gripper_btn.enabled = False
        self._gripper_open = False

        self._ar = _dynamic_control.INVALID_HANDLE

        self._reset_btn = self._window.layout.add_child(omni.kit.ui.Button("Reset"))
        self._reset_btn.set_clicked_fn(self._on_reset)
        self._reset_btn.enabled = False
        self._reset_btn.tooltip = omni.kit.ui.Label("Reset Robot to default position")

        self._settings = omni.kit.settings.get_settings_interface()

        self._settings.set("/persistent/physics/updateToUsd", False)
        self._settings.set("/persistent/physics/useFastCache", True)
        self._settings.set("/persistent/physics/numThreads", 8)

        self._sub_stage_event = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event
        )
        self._termination_criteria = FrameTerminationCriteria(orig_thresh=0.001)

        self._first_step = True
        self._robot = None

        ## unit conversions: RMP is in meters, kit is by default in cm
        self._meters_per_unit = UsdGeom.GetStageMetersPerUnit(self._stage)
        self._units_per_meter = 1.0 / UsdGeom.GetStageMetersPerUnit(self._stage)

    def _on_environment_setup(self, widget):
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
        self._world = World(self._dc, self._mp)

        ## register robot with RMP
        robot_path = "/scene/robot"
        self._robot = Franka(
            self._stage, self._stage.GetPrimAtPath(robot_path), self._dc, self._mp, self._world, default_config
        )

    def _on_target_following(self, widget):
        ## create target
        target_path = "/scene/target"
        if self._stage.GetPrimAtPath(target_path):
            self._following = True
            return
        target_geom = UsdGeom.Sphere.Define(self._stage, target_path)
        offset = Gf.Vec3f(30, 0.0, 30.0)  ## these are in cm
        colors = Gf.Vec3f(1.0, 0, 0)
        target_size = 4
        target_geom.CreateRadiusAttr(target_size)
        target_geom.AddTranslateOp().Set(offset)
        target_geom.CreateDisplayColorAttr().Set([colors])
        self._target_prim = self._stage.GetPrimAtPath(target_path)

        ## start following it
        self._following = True

    def _on_add_obstacle(self, widget):
        ## set ground as an obstacles in RMP
        self._world.register_object(0, "/World/groundPlane/collisionPlane", "ground")
        self._world.make_obstacle(
            "ground", 3, (500 * self._meters_per_unit, 500 * self._meters_per_unit, 10 * self._meters_per_unit)
        )

        ## add a block in Kit
        self._block_path = "/scene/block"
        if self._stage.GetPrimAtPath(self._block_path):
            return
        self._block_geom = UsdGeom.Cube.Define(self._stage, self._block_path)
        offset = Gf.Vec3f(30, -20, 5)
        obstacle_color = Gf.Vec3f(1.0, 1.0, 0)
        size = 10
        self._block_geom.CreateSizeAttr(size)
        self._block_geom.AddTranslateOp().Set(offset)
        self._block_geom.CreateDisplayColorAttr().Set([obstacle_color])
        self._block_prim = self._stage.GetPrimAtPath(self._block_path)

        ## make this obstacle a rigid body with physics and collision properties
        physicsAPI = PhysicsSchema.PhysicsAPI.Apply(self._block_prim)
        physicsAPI.CreateBodyTypeAttr("rigid")
        PhysicsSchema.CollisionAPI.Apply(self._block_prim)

        ## set the block as an obstacle in RMP
        self._world.register_object(0, self._block_path, "block")
        self._world.make_obstacle(
            "block", 3, (size * self._meters_per_unit, size * self._meters_per_unit, size * self._meters_per_unit)
        )

        self._obstacle_on = True

    def _on_toggle_obstacle(self, widget):
        """an obstacle can be temporarily suppressed so that the collision avoidance algorithm ignores it. This can be useful if you need to get very close to an object.
        """
        block_suppressor = self._world.get_object_from_name("block")
        invisible_color = Gf.Vec3f(0.0, 0.0, 1.0)
        obstacle_color = Gf.Vec3f(1.0, 1.0, 0)

        if self._obstacle_on:
            block_suppressor.suppress()
            self._block_geom.GetDisplayColorAttr().Set([invisible_color])
            self._obstacle_on = False
        else:
            block_suppressor.unsuppress()
            self._block_geom.GetDisplayColorAttr().Set([obstacle_color])
            self._obstacle_on = True

    def _on_toggle_gripper(self, widget):
        if self._gripper_open:
            print("closing gripper")
            self._robot.end_effector.gripper.close()
            self._gripper_open = False
        else:
            print("opening gripper")
            self._robot.end_effector.gripper.open()
            self._gripper_open = True

    def _on_grasp_object(self, widget):
        pass

    def _on_editor_step(self, step):
        """This function is called every timestep in the editor
        
        Arguments:
            step (float): elapsed time between steps
        """
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

    def _on_reset(self, widget):
        self._following = False

        # put robot (an articulated prim) in a specific joint configuration
        reset_config = np.array([0.00, -1.3, 0.00, -2.57, 0.00, 2.20, 0.75])
        self._robot.send_config(reset_config)
        self._robot.end_effector.go_local(use_default_config=True, wait_for_target=False)
        self._robot.end_effector.gripper.close()
        self._gripper_open = False

        # put target back (a visual prim) in position
        if self._target:
            self._target_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(30.0, 0.0, 30))

        # put obstacle block (a rigid body prim) back in position
        if self._block_prim:
            start_pose = _dynamic_control.Transform()
            start_pose.p = (30.0, -20.0, 5)
            start_pose.r = (0, 0, 0, 1)
            block_handle = self._dc.get_rigid_body(self._block_path)
            self._dc.set_rigid_body_pose(block_handle, start_pose)

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
            self._add_obstacle_btn.enabled = False
            self._toggle_obstacle_btn.enabled = False
            self._gripper_btn.enabled = False
            self._reset_btn.enabled = False
            self._editor.stop()
            self._stop_tasks()

    def _on_update_ui(self, widget):
        """Callback that updates UI elements every frame
        """
        if self._created:
            self._create_robot_btn.enabled = True
            self._target_following_btn.enabled = False
            self._add_obstacle_btn.enabled = False
            self._toggle_obstacle_btn.enabled = False
            self._gripper_btn.enabled = False
            self._reset_btn.enabled = False
            if self._editor.is_playing():
                self._target_following_btn.enabled = True
                self._target_following_btn.text = "Follow Target"
                self._add_obstacle_btn.enabled = True
                self._gripper_btn.enabled = True
                self._reset_btn.enabled = True
                if self._block_prim:
                    self._toggle_obstacle_btn.enabled = True
                    if self._obstacle_on:
                        self._toggle_obstacle_btn.text = "Press to Suppress Block"
                    else:
                        self._toggle_obstacle_btn.text = "Press to Unsuppress Block"
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
        self._window.set_update_fn(None)
        gc.collect()

    def has_arrived(self):
        """if multiple targets are sent, the later one will overwrite the earlier one. 
            Use this function to check for arrived condition to be met before going to the next target.
        """
        return self._termination_criteria(self._target, self._robot.end_effector.status.current_frame)
