#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import carb
from pxr import Usd, UsdGeom, Sdf, Gf, PhysicsSchema, PhysxSchema
import omni.usd
import gc
import omni.kit.connectionhub
from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server


# Utility function to specify the stage with the z axis as "up"
def set_up_z_axis(stage):
    rootLayer = stage.GetRootLayer()
    rootLayer.SetPermissionToEdit(True)
    with Usd.EditContext(stage, rootLayer):
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


# Specify position of a given prim, reuse any existing transform ops when possible
def set_translate(prim, new_loc):
    properties = prim.GetPropertyNames()
    if "xformOp:translate" in properties:
        translate_attr = prim.GetAttribute("xformOp:translate")
        translate_attr.Set(new_loc)
    elif "xformOp:translation" in properties:
        translation_attr = prim.GetAttribute("xformOp:translate")
        translation_attr.Set(new_loc)
    elif "xformOp:transform" in properties:
        transform_attr = prim.GetAttribute("xformOp:transform")
        matrix = prim.GetAttribute("xformOp:transform").Get()
        matrix.SetTranslateOnly(new_loc)
        transform_attr.Set(matrix)
    else:
        xform = UsdGeom.Xformable(prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        xform_op.Set(Gf.Matrix4d().SetTranslate(new_loc))


# Specify collision group for a prim
def set_collision_group(stage, path, group):
    collisionAPI = PhysicsSchema.CollisionAPI.Get(stage, path)
    if collisionAPI:
        rel = collisionAPI.CreateCollisionGroupRel()
        rel.AddTarget(Sdf.Path(group))


# Specify collision group for franka USD and franka ghost USD
def setCollisionGroupFranka(stage, prim_path, group_path, is_ghost):
    franka_prim = stage.GetPrimAtPath(prim_path)

    for p in Usd.PrimRange(franka_prim):
        set_collision_group(stage, p.GetPath(), group_path)


# Instantiate a solid franka usd in the stage at a given path
def create_solid_franka(stage, env_path, franka_stage, solid_robot, location):
    envPrim = stage.DefinePrim(env_path, "Xform")  # create an empty Xform at the given path
    envPrim.GetReferences().AddReference(franka_stage)  # attach the franka table USD to the given path
    set_translate(envPrim, location)  # set pose of the franka table usd
    setCollisionGroupFranka(stage, env_path + "/Franka/panda", solid_robot, False)  # Set the collision group to solid


# Instantiates a ghost franka in the stage at the given path
def create_ghost_franka(stage, env_path, franka_ghost_usd, ghost_robot, ghost_index):
    ghost_path = env_path + "/Ghost/robot_{}".format(ghost_index)
    ghostPrim = stage.DefinePrim(ghost_path, "Xform")
    ghostPrim.GetReferences().AddReference(franka_ghost_usd)
    set_translate(ghostPrim, Gf.Vec3d(0, 0, 0))
    imageable = UsdGeom.Imageable(ghostPrim)  # Hide the ghost franka when spawned
    if imageable:
        imageable.MakeInvisible()
    # set the ghost collision for this prim
    setCollisionGroupFranka(stage, ghost_path + "/Franka/panda", ghost_robot, True)


# Spawn N blocks given a list of paths in the stage
# All 3 lists must be the same length
def create_blocks(stage, asset_paths, env_paths, poses):
    if (len(asset_paths) != len(env_paths)) and len(asset_paths) != len(poses):
        print("Error: asset paths, env paths and poses must be same length")
        return
    for (asset, path, pose) in zip(*[asset_paths, env_paths, poses]):
        cubePrim = stage.DefinePrim(path, "Xform")
        cubePrim.GetReferences().AddReference(asset)
        set_translate(cubePrim, pose)


# Spawn  rubiks cube usd at specified path
def create_rubiks_cube(stage, asset_path, prim_path, location):
    obstaclePrim = stage.DefinePrim(prim_path, "Xform")
    obstaclePrim.GetReferences().AddReference(asset_path)
    set_translate(obstaclePrim, location)


# Create background stage
def create_background(stage, background_stage):
    background_path = "/background"
    if not stage.GetPrimAtPath(background_path):
        backPrim = stage.DefinePrim(background_path, "Xform")
        backPrim.GetReferences().AddReference(background_stage)
        # Move the stage down -104cm so that the floor is below the table wheels, move in y axis to get light closer
        set_translate(backPrim, Gf.Vec3d(0, -400, -104))


# Set default physics parameters
def setup_physics(stage):
    # Specify gravity
    metersPerUnit = UsdGeom.GetStageMetersPerUnit(stage)
    gravityScale = 9.81 / metersPerUnit
    gravity = Gf.Vec3f(0.0, 0.0, -gravityScale)
    scene = PhysicsSchema.PhysicsScene.Define(stage, "/physics/scene")
    scene.CreateGravityAttr().Set(gravity)

    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physics/scene"))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physics/scene")
    physxSceneAPI.CreatePhysxSceneEnableCCDAttr(True)
    physxSceneAPI.CreatePhysxSceneEnableStabilizationAttr(True)
    physxSceneAPI.CreatePhysxSceneEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreatePhysxSceneBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreatePhysxSceneSolverTypeAttr("TGS")


class Scenario:
    """ Defines a block stacking scenario

    Scenarios define the life cycle within kit and handle init, startup, shutdown etc.
    """

    def __init__(self, editor, dc, mp):
        self._editor = editor  # Reference to the Kit editor
        self._stage = omni.usd.get_context().get_stage()  # Reference to the current USD stage
        self._dc = dc  # Reference to the dynamic control plugin
        self._mp = mp  # Reference to the motion planning plugin
        self._domains = []  # Contains instances of environment
        self._obstacles = []  # Containts references to any obstacles in the scenario
        self._executor = None  # Contains the thread pool used to run tasks
        self._created = False  # Is the robot created or not
        self._running = False  # Is the task running or not

    # Cleanup scenario objects when deleted, force garbage collection
    def __del__(self):
        self.robot_created = False
        self._domains = []
        self._obstacles = []
        self._executor = None
        gc.collect()

    # Funtion called when block poses are reset
    def reset_blocks(self, *args):
        pass

    # Stop tasks in the scenario if any
    def stop_tasks(self, *args):
        self._running = False
        pass

    # Step the scenario, can be used to update things in the scenario per frame
    def step(self, step):
        pass

    # Create frana USD objects
    def create_franka(self, *args):
        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        self.asset_path = nucleus_server + "/Isaac"

        # USD paths loaded by scenarios
        self.franka_table_usd = self.asset_path + "/Samples/Leonardo/Stage/franka_block_stacking.usd"
        self.franka_ghost_usd = self.asset_path + "/Samples/Leonardo/Robots/franka_ghost.usd"
        self.background_usd = self.asset_path + "/Environments/Grid/gridroom_curved.usd"
        self.rubiks_cube_usd = self.asset_path + "/Props/Rubiks_Cube/rubiks_cube.usd"
        self.red_cube_usd = self.asset_path + "/Props/Blocks/red_block.usd"
        self.yellow_cube_usd = self.asset_path + "/Props/Blocks/yellow_block.usd"
        self.green_cube_usd = self.asset_path + "/Props/Blocks/green_block.usd"
        self.blue_cube_usd = self.asset_path + "/Props/Blocks/blue_block.usd"

        self._created = True
        self._stage = omni.usd.get_context().get_stage()
        set_up_z_axis(self._stage)
        self.stop_tasks()
        pass

    # Connect franka controller to usd assets
    def register_assets(self, *args):
        pass

    # Task to be performed for a given robot
    def task(self, domain):
        pass

    # Perform all tasks in scenario if multiple robots are present
    def perform_tasks(self, *args):
        self._running = True
        pass

    # Return if the franka was already created
    def is_created(self):
        return self._created
