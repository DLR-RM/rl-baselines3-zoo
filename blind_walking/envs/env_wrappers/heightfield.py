"""
CODE BASED ON EXAMPLE FROM:
@misc{coumans2017pybullet,
  title={Pybullet, a python module for physics simulation in robotics, games and machine learning},
  author={Coumans, Erwin and Bai, Yunfei},
  url={www.pybullet.org},
  year={2017},
}

Example: heightfield.py
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/heightfield.py
"""

import pybullet as p
import pybullet_data as pd
import math
import time

textureId = -1

useProgrammatic = 0
useTerrainFromPNG = 1
useDeepLocoCSV = 2
updateHeightfield = False

heightfieldSource = useProgrammatic
numHeightfieldRows = 256
numHeightfieldColumns = 256
import random
random.seed(10)


class HeightField():
    def __init__(self):
        self.hf_id = 0
        self.terrainShape = 0
        self.heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

    def _generate_field(self, env, heightPerturbationRange=0.08, friction=1.0):
        env.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 0)
        heightPerturbationRange = heightPerturbationRange
        if heightfieldSource == useProgrammatic:
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = random.uniform(0, heightPerturbationRange)
                    self.heightfieldData[2 * i +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + (2 * j + 1) *
                                         numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                                         numHeightfieldRows] = height

            terrainShape = env.pybullet_client.createCollisionShape(
                shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.07, .07, 1.6],
                heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                heightfieldData=self.heightfieldData,
                numHeightfieldRows=numHeightfieldRows,
                numHeightfieldColumns=numHeightfieldColumns)
            terrain = env.pybullet_client.createMultiBody(0, terrainShape)
            env.pybullet_client.resetBasePositionAndOrientation(
                terrain, [0, 0, 0.0], [0, 0, 0, 1])
            env.pybullet_client.changeDynamics(terrain,
                                               -1,
                                               lateralFriction=friction)

        if heightfieldSource == useDeepLocoCSV:
            terrainShape = env.pybullet_client.createCollisionShape(
                shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.5, .5, 2.5],
                fileName="heightmaps/ground0.txt",
                heightfieldTextureScaling=128)
            terrain = env.pybullet_client.createMultiBody(0, terrainShape)
            env.pybullet_client.resetBasePositionAndOrientation(
                terrain, [0, 0, 0], [0, 0, 0, 1])
            env.pybullet_client.changeDynamics(terrain,
                                               -1,
                                               lateralFriction=friction)

        if heightfieldSource == useTerrainFromPNG:
            terrainShape = env.pybullet_client.createCollisionShape(
                shapeType=env.pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.05, .05, 1.8],
                fileName="heightmaps/wm_height_out.png")
            textureId = env.pybullet_client.loadTexture(
                "heightmaps/gimp_overlay_out.png")
            terrain = env.pybullet_client.createMultiBody(0, terrainShape)
            env.pybullet_client.changeVisualShape(terrain,
                                                  -1,
                                                  textureUniqueId=textureId)
            env.pybullet_client.resetBasePositionAndOrientation(
                terrain, [0, 0, 0.1], [0, 0, 0, 1])
            env.pybullet_client.changeDynamics(terrain,
                                               -1,
                                               lateralFriction=friction)

        self.hf_id = terrainShape
        self.terrainShape = terrainShape
        print("TERRAIN SHAPE: {}".format(terrainShape))

        env.pybullet_client.changeVisualShape(terrain,
                                              -1,
                                              rgbaColor=[1, 1, 1, 1])

        env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 1)

    def UpdateHeightField(self, heightPerturbationRange=0.08):
        if heightfieldSource == useProgrammatic:
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = random.uniform(
                        0, heightPerturbationRange)  # +math.sin(time.time())
                    self.heightfieldData[2 * i +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 +
                                         2 * j * numHeightfieldRows] = height
                    self.heightfieldData[2 * i + (2 * j + 1) *
                                         numHeightfieldRows] = height
                    self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                                         numHeightfieldRows] = height
            #GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
            #GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
            #flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            flags = 0
            self.terrainShape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                flags=flags,
                meshScale=[.05, .05, 1],
                heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                heightfieldData=self.heightfieldData,
                numHeightfieldRows=numHeightfieldRows,
                numHeightfieldColumns=numHeightfieldColumns,
                replaceHeightfieldIndex=self.terrainShape)
