import pybullet as p
import pybullet_data as pd

import random
random.seed(10)


class CarryMass():
    def __init__(self):
        self.cm_id = 0

    def _generate_mass(self, env, mass=1, mass_pos=[0,0,0]):
        env.pybullet_client.setAdditionalSearchPath(pd.getDataPath())

        env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 0)

        # create mass
        carrymassShape = env.pybullet_client.createCollisionShape(
            shapeType=env.pybullet_client.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.05])
        carrymass = env.pybullet_client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=carrymassShape)
        # fix mass onto robot
        carrymassFixed = env.pybullet_client.createConstraint(
            parentBodyUniqueId=1,
            parentLinkIndex=-1,
            childBodyUniqueId=carrymass,
            childLinkIndex=-1,
            jointType=env.pybullet_client.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=mass_pos,
            childFramePosition=[0,0,0]
        )
     
        env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 1)
