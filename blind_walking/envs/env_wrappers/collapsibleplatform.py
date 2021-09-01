import pybullet as p
import pybullet_data as pd

import os
import numpy as np
import pyquaternion
import time

import random
random.seed(10)

''' Collapsible Platform Env types
Using _generate_field():
Case 1: Front right feet damping platform (Height above ground 0.50m)
Case 2: Front left feed damping platform (Height above ground 0.50m)
Usage: 
self.cp._generate_field(self,
                        case=1,
                        sElasticStiffness=8,
                        sDampingStiffness=1,
                        texture='water_texture' # or 'grass_texture'
                    )

Using _generate_soft_env():
Random soft deformable platform (without damping) (Height above ground = resolution / 2 + plane_z)
Usage:
self.cp._generate_soft_env(self,
                            testing_area_x=5,
                            testing_area_y=3,
                            resolution=0.2,
                            clearance_area_x=2,
                            p_solid_floor=0.6,
                            p_collapse_floor=0.2
                        )

'''

class CollapsiblePlatform():
    def __init__(self):
        self.cp_id = 0
        self.textureId = None
        self.color = 'G'
        
        # Store simulation environment tiles id and default values
        self.platform = []
        self.damping_platform = []
        self.collapsible_platform = []
        self.damping_tile_pos = []
        self.collapsible_tile_pos = []
        self.sElasticStiffness = 8
        self.sDampingStiffness = 1
        # Generate map matrix information structure (for generate_soft_env)
        self.map_mat = {
            'obj_id':[],'color':[],'basePosition':[],'mass':[],
            'sElasticStiff':[],'sDampingStiff':[], 'collapsibility':[]
        }

    def _reset_field_collapsible(self, env):
        for index, blockId in enumerate(self.damping_platform):
            env.pybullet_client.resetBasePositionAndOrientation(
                blockId, self.damping_tile_pos[index], [0, 0, 0, 1])
            env.pybullet_client.resetBaseVelocity(blockId, [0, 0, 0], [0, 0, 0])
            if self.textureId:
                env.pybullet_client.changeVisualShape(blockId, -1, textureUniqueId=self.textureId)
                env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
            elif self.color == 'O':
                env.pybullet_client.changeVisualShape(blockId, -1, rgbaColor=[1,0.5,0,1], flags=0)
                env.pybullet_client.changeDynamics(blockId, -1, mass=0.25)
            elif self.color == 'R':
                env.pybullet_client.changeVisualShape(blockId, -1, rgbaColor=[1,0.25,0,1], flags=0)
                env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
        for index, blockId in enumerate(self.collapsible_platform):
            print(f'{index}, {blockId}')
            env.pybullet_client.removeBody(blockId)
            del self.collapsible_platform[index]
            regenerated_body = env.pybullet_client.loadSoftBody(
                "cube.obj", basePosition = self.collapsible_tile_pos[index], scale = 0.25, mass = 1.,
                useNeoHookean = 0, useBendingSprings=1,useMassSpring=1,
                springElasticStiffness=self.sElasticStiffness, springDampingStiffness=self.sDampingStiffness,
                springDampingAllDirections = 1, collisionMargin=0.01,
                useSelfCollision = 1, frictionCoeff = .5, useFaceContact=1)
            self.collapsible_platform = np.insert(self.collapsible_platform, index, regenerated_body, axis=0)
            if self.textureId:
                env.pybullet_client.changeVisualShape(regenerated_body, -1, textureUniqueId=self.textureId)
            elif self.color == 'O':
                env.pybullet_client.changeVisualShape(regenerated_body, -1, rgbaColor=[1,0.5,0,1], flags=0)
            elif self.color == 'R':
                env.pybullet_client.changeVisualShape(regenerated_body, -1, rgbaColor=[1,0.25,0,1], flags=0)
            # Anchor Soft Body at the bottom 4 corners
            # ground anchor on vertices 4,5,6,7.
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 4, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 5, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 6, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(regenerated_body, 7, -1, -1)

    def _generate_field(self, env, case=1, sElasticStiffness=8, sDampingStiffness=1, texture=None):
        # env.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        # env.pybullet_client.configureDebugVisualizer(
        #     env.pybullet_client.COV_ENABLE_RENDERING, 0)
        platform = []
        damping_platform = []
        collapsible_platform = []
        damping_tile_pos = []
        collapsible_tile_pos = []
        self.sElasticStiffness = sElasticStiffness
        self.sDampingStiffness = sDampingStiffness

        if (sElasticStiffness + sDampingStiffness) <= 41:
            self.color = 'R'
            # collapsibility = 0.0
        elif 41 < (sElasticStiffness + sDampingStiffness) <= 179:
            self.color = 'O'
            # collapsibility = 0.50

        if texture:
            self.textureId = env.pybullet_client.loadTexture(
                'blind_walking/envs/env_wrappers/collapsible_platform/%s.png' %texture)

        if case == 1: # front right feet soft platform
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.125,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.125,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.125,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.125,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.375,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.365,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,0.135,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.875,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.885,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1.125,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1.125,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))

            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.5,-0.25,0.0],
                globalScaling = 1.0,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.5,0.25,0.0],
                globalScaling = 1.0,
                useMaximalCoordinates = True))

            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.250,-0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1,-0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.250,0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1,0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))

            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,0.625,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,0.375,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,-0.625,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,-0.385,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))

            # Damping Platform
            damping_platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_tile.urdf',
                [0.625,-0.125,0.3875],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            damping_tile_pos.append([0.625,-0.125,0.3875])
            collapsible_platform.append(env.pybullet_client.loadSoftBody(
                "cube.obj", basePosition = [0.625,-0.125,0.150], scale = 0.25, mass = 1., 
                useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, 
                springElasticStiffness=sElasticStiffness, springDampingStiffness=sDampingStiffness,
                springDampingAllDirections = 1, collisionMargin=0.01, 
                useSelfCollision = 1, frictionCoeff = .5, useFaceContact=1))
            collapsible_tile_pos.append([0.625,-0.125,0.150])

        if case == 2: # front left feet soft platform
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.125,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.125,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.125,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.125,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.365,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.375,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,-0.135,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.885,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.875,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1.125,0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1.125,-0.125,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))

            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.5,-0.25,0.0],
                globalScaling = 1.0,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [-0.5,0.25,0.0],
                globalScaling = 1.0,
                useMaximalCoordinates = True))
            
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.250,-0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1,-0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.250,0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [1,0.50,0.250],
                globalScaling = 0.5,
                useMaximalCoordinates = True))

            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,0.625,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,0.385,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,-0.625,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                [0.625,-0.375,0.375],
                globalScaling = 0.25,
                useMaximalCoordinates = True))

            # Damping Platform
            damping_platform.append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_tile.urdf',
                [0.625,0.125,0.3875],
                globalScaling = 0.25,
                useMaximalCoordinates = True))
            damping_tile_pos.append([0.625,0.125,0.3875])
            collapsible_platform.append(env.pybullet_client.loadSoftBody(
                "cube.obj", basePosition = [0.625,0.125,0.150], scale = 0.25, mass = 1., 
                useNeoHookean = 0, useBendingSprings=1,useMassSpring=1, 
                springElasticStiffness=sElasticStiffness, springDampingStiffness=sDampingStiffness,
                springDampingAllDirections = 1, collisionMargin=0.01, 
                useSelfCollision = 1, frictionCoeff = .5, useFaceContact=1))
            collapsible_tile_pos.append([0.625,0.125,0.150])

        # Assign color
        if case in [1,2]:
            for index, blockId in enumerate(damping_platform):
                if texture:
                    env.pybullet_client.changeVisualShape(blockId, -1, textureUniqueId=self.textureId)
                    env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
                elif self.color == 'O':
                    env.pybullet_client.changeVisualShape(blockId, -1, rgbaColor=[1,0.5,0,1], flags=0)
                    env.pybullet_client.changeDynamics(blockId, -1, mass=0.25)
                elif self.color == 'R':
                    env.pybullet_client.changeVisualShape(blockId, -1, rgbaColor=[1,0.25,0,1], flags=0)
                    env.pybullet_client.changeDynamics(blockId, -1, mass=0.07)
            for index, blockId in enumerate(collapsible_platform):
                if texture:
                    env.pybullet_client.changeVisualShape(blockId, -1, textureUniqueId=self.textureId)
                elif self.color == 'O':
                    env.pybullet_client.changeVisualShape(blockId, -1, rgbaColor=[1,0.5,0,1], flags=0)
                elif self.color == 'R':
                    env.pybullet_client.changeVisualShape(blockId, -1, rgbaColor=[1,0.25,0,1], flags=0)
                # Anchor Soft Body at the bottom 4 corners
                env.pybullet_client.createSoftBodyAnchor(blockId, 4, -1, -1) # ground anchor on vertices 4,5,6,7. 
                env.pybullet_client.createSoftBodyAnchor(blockId, 5, -1, -1)
                env.pybullet_client.createSoftBodyAnchor(blockId, 6, -1, -1)
                env.pybullet_client.createSoftBodyAnchor(blockId, 7, -1, -1)
        print("TERRAIN TYPE: Collapsible Platform - Case {}".format(case))

        env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 1)

        self.platform = platform
        self.damping_platform = damping_platform
        self.collapsible_platform = collapsible_platform
        self.damping_tile_pos = damping_tile_pos
        self.collapsible_tile_pos = collapsible_tile_pos
        # return platform # list of platform id
        # For case 1 & 2: the last id is the soft platform, the 2nd last is the platform on the soft tile)

    def _generate_soft_env(self, env,
                           testing_area_x=5, testing_area_y=3,
                           resolution=0.2, clearance_area_x=2,
                           p_solid_floor=0.6, p_collapse_floor=0.2):
      # map_colour_matrix -  __  TOP VIEW
      #                     |       -----> +y (robot left)
      #                             |           number of tiles in x-direction = testing_area_x / 0.2
      #                             |           number of tiles in y-direction = testing_area_y / 0.2
      #                            \|/ (robot forward)
      #                            +x        _|
      # spring_elastic_sum_damping_stiffness_range - 
      #                                               red (R): 41
      #                                               orange (O): 52 - 179
      #                                               green (G): N/A
      # recommended minimum_testing_area_x = 5.0 metres
      # recommended minimum_testing_area_y = 3.0 metres
      # recommended resolution = 0.2 metre
      # recommended clearance_area_x = 2.0 metre
      # recommended probability of solid ground (not counting clearance area) = 0.6
      # recommended probability of collapsible ground = 0.2

      # Calculate probability of soft floor (2 types) in map
      p_other_floor = (1 - p_solid_floor - p_collapse_floor)
      max_body_limit = 32000
      plane_z = 0
      # Since max_body_limit is 32000, for resolution = 0.1, testing_area_y = 3, max. testing_area_x = 106. 

      # Check if map building boundary condition violated
      while True:
        try:
          assert testing_area_x >= 3.0
          assert testing_area_y >= 2.0
          assert resolution < testing_area_x and resolution < testing_area_y
          n_x_grid = int(np.floor(testing_area_x / resolution))
          n_y_grid = int(np.floor(testing_area_y / resolution))
          assert n_x_grid * n_y_grid < 32000 - 2 # assuming 2 other bodies in map, plane and robot
          n_x_clear_grid = np.floor(clearance_area_x / resolution) # No. of clearance grids
          break
        except:
          print("Invalid input. Please check recommended testing area size and resolution.")
      
      # Generate map matrix information structure
      map_mat = {'obj_id':[],'color':[],'basePosition':[],'mass':[],
                 'sElasticStiff':[],'sDampingStiff':[], 'collapsibility':[]}
      
      # Generate colour map matrix
      while True:
        print("Generating map")
        color_count = {'R':0,'O':0,'G':0}
        map_mat['color'] = []
        for x in range(n_x_grid):
          map_mat['color'].append([])
          for y in range(n_y_grid):
            if x+1 <= n_x_clear_grid: # check if still within clearance area
              map_mat['color'][x].append('G')
            else: # assign 
              map_mat['color'][x].append(np.random.choice(
                  [str(key) for key in color_count], 1,
                  p=[p_collapse_floor, p_other_floor, p_solid_floor])[0])
            color_count[map_mat['color'][x][y]] += 1
        map_mat['color'][-1][-1] = 'FG'
        # Check if sufficient collapsible ground
        if color_count['R'] < p_collapse_floor * (n_x_grid - n_x_clear_grid) * n_y_grid:
          continue
        
        # Verify if map is valid
        valid = False

        # Depth-first search (DFS) to check that it's a valid path.
        def is_valid(res, n_x_grid, n_y_grid):
            frontier, discovered = [], set()
            frontier.append((0, 0))
            while frontier:
                r, c = frontier.pop()
                if not (r, c) in discovered:
                    discovered.add((r, c))
                    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    for x, y in directions:
                        r_new = r + x
                        c_new = c + y
                        if r_new < 0 or r_new >= n_x_grid or c_new < 0 or c_new >= n_y_grid:
                            continue
                        if res[r_new][c_new] == 'FG':
                            return True
                        if (res[r_new][c_new] != 'R'):
                            frontier.append((r_new, c_new))
            return False

        valid = is_valid(map_mat['color'], n_x_grid, n_y_grid)
        if valid:
          break
        print("Invalid map, regenerating.")

      map_mat['color'][-1][-1] = 'G'

      # Generate all other information of each tile and load platform & colour-code. 
      map_mat['basePosition'] = []
      start_base_pos_x = clearance_area_x / 2 * (-1)
      start_base_pos_y = testing_area_y / 2 * (-1)
      pos_x = start_base_pos_x
      pos_y = start_base_pos_y
      pos_z = resolution / 2 + plane_z
      for x in range(n_x_grid):
        map_mat['basePosition'].append([])
        map_mat['mass'].append([])
        map_mat['sElasticStiff'].append([])
        map_mat['sDampingStiff'].append([])
        map_mat['obj_id'].append([])
        map_mat['collapsibility'].append([])
        for y in range(n_y_grid):
          map_mat['basePosition'][x].append([pos_x, pos_y, pos_z])
          if map_mat['color'][x][y] == 'G':
            map_mat['mass'][x].append(0) # static
            map_mat['sElasticStiff'][x].append(520.5)
            map_mat['sDampingStiff'][x].append(520.5)
            map_mat['obj_id'][x].append(env.pybullet_client.loadURDF(
                'blind_walking/envs/env_wrappers/collapsible_platform/cube_platform.urdf',
                map_mat['basePosition'][x][y],
                globalScaling = resolution,
                useMaximalCoordinates = True))
          elif map_mat['color'][x][y] == 'Y':
            map_mat['mass'][x].append(np.random.uniform(1,2))
            map_mat['sElasticStiff'][x].append(np.random.randint(100,500))
            map_mat['sDampingStiff'][x].append(np.random.randint(100,500))
            map_mat['obj_id'][x].append(env.pybullet_client.loadSoftBody(
                "cube.obj", basePosition = map_mat['basePosition'][x][y], 
                scale = resolution, mass = map_mat['mass'][x][y], 
                useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, 
                springElasticStiffness=map_mat['sElasticStiff'][x][y], 
                springDampingStiffness=map_mat['sDampingStiff'][x][y], 
                springDampingAllDirections = 1, collisionMargin=0.01, 
                useSelfCollision = 1, frictionCoeff = .5, useFaceContact=1))
            env.pybullet_client.changeVisualShape(
                map_mat['obj_id'][x][y], -1, rgbaColor=[1,1,0,1], flags=0) # yellow - Y
          elif map_mat['color'][x][y] == 'O':
            map_mat['mass'][x].append(1)
            map_mat['sElasticStiff'][x].append(np.random.randint(41,99))
            map_mat['sDampingStiff'][x].append(np.random.randint(10,80))
            map_mat['obj_id'][x].append(env.pybullet_client.loadSoftBody(
                "cube.obj", basePosition = map_mat['basePosition'][x][y], 
                scale = resolution, mass = map_mat['mass'][x][y], 
                useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, 
                springElasticStiffness=map_mat['sElasticStiff'][x][y], 
                springDampingStiffness=map_mat['sDampingStiff'][x][y], 
                springDampingAllDirections = 1, collisionMargin=0.01, 
                useSelfCollision = 1, frictionCoeff = .5, useFaceContact=1))
            env.pybullet_client.changeVisualShape(
                map_mat['obj_id'][x][y], -1, rgbaColor=[1,0.5,0,1], flags=0) # orange - O
          elif map_mat['color'][x][y] == 'R':
            map_mat['mass'][x].append(1)
            map_mat['sElasticStiff'][x].append(40)
            map_mat['sDampingStiff'][x].append(1)
            map_mat['obj_id'][x].append(env.pybullet_client.loadSoftBody(
                "cube.obj", basePosition = map_mat['basePosition'][x][y], 
                scale = resolution, mass = map_mat['mass'][x][y], 
                useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, 
                springElasticStiffness=map_mat['sElasticStiff'][x][y], 
                springDampingStiffness=map_mat['sDampingStiff'][x][y], 
                springDampingAllDirections = 1, collisionMargin=0.01, 
                useSelfCollision = 1, frictionCoeff = .5, useFaceContact=1))
            env.pybullet_client.changeVisualShape(
                map_mat['obj_id'][x][y], -1, rgbaColor=[1,0.25,0,1], flags=0) # red - R
          if map_mat['color'][x][y] != 'G':
            # ground anchor on vertices 4,5,6,7. 
            env.pybullet_client.createSoftBodyAnchor(map_mat['obj_id'][x][y], 4, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(map_mat['obj_id'][x][y], 5, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(map_mat['obj_id'][x][y], 6, -1, -1)
            env.pybullet_client.createSoftBodyAnchor(map_mat['obj_id'][x][y], 7, -1, -1)
          map_mat['collapsibility'][x].append(
              1041 - (map_mat['sElasticStiff'][x][y] + map_mat['sDampingStiff'][x][y]))
          pos_y += resolution
        pos_x += resolution
        pos_y = start_base_pos_y
      print("Total number of grids generated: ", map_mat['obj_id'][-1][-1] - map_mat['obj_id'][1][1] + 1)
      print("Map generated.")
      print("TERRAIN TYPE: Collapsible Platform - Random")

      env.pybullet_client.configureDebugVisualizer(
            env.pybullet_client.COV_ENABLE_RENDERING, 1)
      self.map_mat = map_mat
      # return map_mat
