
## Performance of trained agents

Final performance of the trained agents can be found in the table below.
This was computed by running `python -m rl_zoo3.benchmark`:
it runs the trained agent (trained on `n_timesteps`) for `eval_timesteps` and then reports the mean episode reward
during this evaluation.

It uses the deterministic policy except for Atari games.

You can view each model card (it includes video and hyperparameters)
on our Huggingface page: https://huggingface.co/sb3

*NOTE: this is not a quantitative benchmark as it corresponds to only one run
(cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)).
This benchmark is meant to check algorithm (maximal) performance, find potential bugs
and also allow users to have access to pretrained agents.*

"M" stands for Million (1e6)

|  algo  |            env_id             |mean_reward|std_reward|n_timesteps|eval_timesteps|eval_episodes|
|--------|-------------------------------|----------:|---------:|-----------|-------------:|------------:|
|a2c     |Acrobot-v1                     |    -83.353|    17.213|500k       |        149979|         1778|
|a2c     |Ant-v3                         |    -44.023|    63.206|1M         |        149469|          761|
|a2c     |AntBulletEnv-v0                |   2497.147|    37.359|2M         |        150000|          150|
|a2c     |AsteroidsNoFrameskip-v4        |   1286.550|   423.750|10M        |        614138|          258|
|a2c     |BeamRiderNoFrameskip-v4        |   2890.298|  1379.137|10M        |        591104|           47|
|a2c     |BipedalWalker-v3               |    299.754|    23.459|5M         |        149287|          208|
|a2c     |BipedalWalkerHardcore-v3       |     96.171|   122.943|200M       |        149704|          113|
|a2c     |BreakoutNoFrameskip-v4         |    279.793|   122.177|10M        |        604115|           82|
|a2c     |CartPole-v1                    |    500.000|     0.000|500k       |        150000|          300|
|a2c     |EnduroNoFrameskip-v4           |      0.000|     0.000|10M        |        599040|           45|
|a2c     |HalfCheetah-v3                 |   3041.174|   157.265|1M         |        150000|          150|
|a2c     |HalfCheetahBulletEnv-v0        |   2107.384|    36.008|2M         |        150000|          150|
|a2c     |Hopper-v3                      |    733.454|   376.574|1M         |        149987|          580|
|a2c     |HopperBulletEnv-v0             |    815.355|   313.798|2M         |        149541|          254|
|a2c     |Humanoid-v3                    |    388.321|    92.652|2M         |        149942|         1944|
|a2c     |LunarLander-v2                 |    155.751|    80.419|200k       |        149443|          297|
|a2c     |LunarLanderContinuous-v2       |     84.225|   145.906|5M         |        149305|          256|
|a2c     |MountainCar-v0                 |   -111.263|    24.087|1M         |        149982|         1348|
|a2c     |MountainCarContinuous-v0       |     91.166|     0.255|100k       |        149923|         1659|
|a2c     |MsPacmanNoFrameskip-v4         |   1671.730|   612.918|10M        |        602450|          185|
|a2c     |Pendulum-v1                    |   -162.965|   103.210|1M         |        150000|          750|
|a2c     |PongNoFrameskip-v4             |     17.292|     3.214|10M        |        594910|           65|
|a2c     |QbertNoFrameskip-v4            |   3882.345|  1223.327|10M        |        610670|          194|
|a2c     |ReacherBulletEnv-v0            |     14.968|    10.978|2M         |        150000|         1000|
|a2c     |RoadRunnerNoFrameskip-v4       |  31671.512|  6364.085|10M        |        606710|          172|
|a2c     |SeaquestNoFrameskip-v4         |   1721.493|   105.339|10M        |        599691|           67|
|a2c     |SpaceInvadersNoFrameskip-v4    |    627.160|   201.974|10M        |        604848|          162|
|a2c     |Swimmer-v3                     |    200.627|     2.544|1M         |        150000|          150|
|a2c     |Walker2DBulletEnv-v0           |    858.209|   333.116|2M         |        149156|          173|
|a2c     |Walker2d-v3                    |    581.835|   127.597|1M         |        149782|          593|
|ars     |Acrobot-v1                     |    -82.884|    23.825|500k       |        149985|         1788|
|ars     |Ant-v3                         |   2333.773|    20.597|75M        |        150000|          150|
|ars     |CartPole-v1                    |    500.000|     0.000|50k        |        150000|          300|
|ars     |HalfCheetah-v3                 |   4815.192|  1340.752|12M        |        150000|          150|
|ars     |Hopper-v3                      |   3343.919|     5.730|7M         |        150000|          150|
|ars     |LunarLanderContinuous-v2       |    167.959|   147.071|2M         |        149883|          562|
|ars     |MountainCar-v0                 |   -122.000|    33.456|500k       |        149938|         1229|
|ars     |MountainCarContinuous-v0       |     96.672|     0.784|500k       |        149990|          621|
|ars     |Pendulum-v1                    |   -212.540|   160.444|2M         |        150000|          750|
|ars     |Swimmer-v3                     |    355.267|    12.796|2M         |        150000|          150|
|ars     |Walker2d-v3                    |   2993.582|   166.289|75M        |        149821|          152|
|ddpg    |AntBulletEnv-v0                |   2399.147|    75.410|1M         |        150000|          150|
|ddpg    |BipedalWalker-v3               |    197.486|   141.580|1M         |        149237|          227|
|ddpg    |HalfCheetahBulletEnv-v0        |   2078.325|   208.379|1M         |        150000|          150|
|ddpg    |HopperBulletEnv-v0             |   1157.065|   448.695|1M         |        149565|          346|
|ddpg    |LunarLanderContinuous-v2       |    230.217|    92.372|300k       |        149862|          556|
|ddpg    |MountainCarContinuous-v0       |     93.512|     0.048|300k       |        149965|         2260|
|ddpg    |Pendulum-v1                    |   -152.099|    94.282|20k        |        150000|          750|
|ddpg    |ReacherBulletEnv-v0            |     15.582|     9.606|300k       |        150000|         1000|
|ddpg    |Walker2DBulletEnv-v0           |   1387.591|   736.955|1M         |        149051|          208|
|dqn     |Acrobot-v1                     |    -76.639|    11.752|100k       |        149998|         1932|
|dqn     |AsteroidsNoFrameskip-v4        |    782.687|   259.247|10M        |        607962|          134|
|dqn     |BeamRiderNoFrameskip-v4        |   4295.946|  1790.458|10M        |        600832|           37|
|dqn     |BreakoutNoFrameskip-v4         |    358.327|    61.981|10M        |        601461|           55|
|dqn     |CartPole-v1                    |    500.000|     0.000|50k        |        150000|          300|
|dqn     |EnduroNoFrameskip-v4           |    830.929|   194.544|10M        |        599040|           14|
|dqn     |LunarLander-v2                 |    154.382|    79.241|100k       |        149373|          200|
|dqn     |MountainCar-v0                 |   -100.849|     9.925|120k       |        149962|         1487|
|dqn     |MsPacmanNoFrameskip-v4         |   2682.929|   492.567|10M        |        599952|          140|
|dqn     |PongNoFrameskip-v4             |     20.602|     0.613|10M        |        598998|           88|
|dqn     |QbertNoFrameskip-v4            |   9496.774|  5399.633|10M        |        605844|          124|
|dqn     |RoadRunnerNoFrameskip-v4       |  40396.350|  7069.131|10M        |        603257|          137|
|dqn     |SeaquestNoFrameskip-v4         |   2000.290|   606.644|10M        |        599505|           69|
|dqn     |SpaceInvadersNoFrameskip-v4    |    622.742|   201.564|10M        |        604311|          155|
|ppo     |Acrobot-v1                     |    -73.506|    18.201|1M         |        149979|         2013|
|ppo     |Ant-v3                         |   1327.158|   451.577|1M         |        149572|          175|
|ppo     |AntBulletEnv-v0                |   2865.922|    56.468|2M         |        150000|          150|
|ppo     |AsteroidsNoFrameskip-v4        |   2156.174|   744.640|10M        |        602092|          149|
|ppo     |BeamRiderNoFrameskip-v4        |   3397.000|  1662.368|10M        |        598926|           46|
|ppo     |BipedalWalker-v3               |    287.939|     2.448|5M         |        149589|          123|
|ppo     |BipedalWalkerHardcore-v3       |    122.374|   117.605|100M       |        148036|          105|
|ppo     |BreakoutNoFrameskip-v4         |    398.033|    33.328|10M        |        600418|           60|
|ppo     |CarRacing-v0                   |    153.938|    74.465|4M         |        179000|          179|
|ppo     |CartPole-v1                    |    500.000|     0.000|100k       |        150000|          300|
|ppo     |EnduroNoFrameskip-v4           |    996.364|   176.090|10M        |        572416|           11|
|ppo     |HalfCheetah-v3                 |   5819.099|   663.530|1M         |        150000|          150|
|ppo     |HalfCheetahBulletEnv-v0        |   2924.721|    64.465|2M         |        150000|          150|
|ppo     |Hopper-v3                      |   2410.435|    10.026|1M         |        150000|          150|
|ppo     |HopperBulletEnv-v0             |   2575.054|   223.301|2M         |        149094|          152|
|ppo     |LunarLander-v2                 |    242.119|    31.823|1M         |        149636|          369|
|ppo     |LunarLanderContinuous-v2       |    270.863|    32.072|1M         |        149956|          526|
|ppo     |MiniGrid-DoorKey-5x5-v0        |      0.964|     0.006|100k       |        149994|        14956|
|ppo     |MiniGrid-Empty-Random-5x5-v0   |      0.968|     0.013|100k       |        149999|        41965|
|ppo     |MiniGrid-FourRooms-v0          |      0.573|     0.311|5M         |        149927|         3270|
|ppo     |MiniGrid-GoToDoor-5x5-v0       |      0.525|     0.476|5M         |        149991|        16918|
|ppo     |MiniGrid-KeyCorridorS3R1-v0    |      0.949|     0.011|500k       |        149990|         9733|
|ppo     |MiniGrid-LockedRoom-v0         |      0.000|     0.000|10M        |        149910|          789|
|ppo     |MiniGrid-MultiRoom-N4-S5-v0    |      0.000|     0.000|10M        |        150000|         1875|
|ppo     |MiniGrid-ObstructedMaze-2Dlh-v0|      0.000|     0.000|10M        |        149760|          260|
|ppo     |MiniGrid-PutNear-6x6-N2-v0     |      0.612|     0.295|10M        |        149998|        14170|
|ppo     |MiniGrid-RedBlueDoors-6x6-v0   |      0.860|     0.313|1M         |        149997|         7846|
|ppo     |MiniGrid-Unlock-v0             |      0.963|     0.016|100k       |        149999|        12646|
|ppo     |MountainCar-v0                 |   -110.423|    19.473|1M         |        149954|         1358|
|ppo     |MountainCarContinuous-v0       |     88.343|     2.572|20k        |        149983|          633|
|ppo     |MsPacmanNoFrameskip-v4         |   1754.356|   172.783|10M        |        600822|          163|
|ppo     |Pendulum-v1                    |   -172.225|   104.159|100k       |        150000|          750|
|ppo     |PongNoFrameskip-v4             |     20.989|     0.105|10M        |        599902|           90|
|ppo     |QbertNoFrameskip-v4            |  15627.108|  3313.538|10M        |        600248|           83|
|ppo     |ReacherBulletEnv-v0            |     17.091|    11.048|1M         |        150000|         1000|
|ppo     |RoadRunnerNoFrameskip-v4       |  40680.645|  6675.058|10M        |        605786|          155|
|ppo     |SeaquestNoFrameskip-v4         |   1783.636|    34.096|10M        |        598243|           66|
|ppo     |SpaceInvadersNoFrameskip-v4    |    960.331|   425.355|10M        |        603771|          136|
|ppo     |Swimmer-v3                     |    281.561|     9.671|1M         |        150000|          150|
|ppo     |Walker2DBulletEnv-v0           |   2109.992|    13.899|2M         |        150000|          150|
|ppo     |Walker2d-v3                    |   3478.798|   821.708|1M         |        149343|          171|
|ppo_lstm|CarRacing-v0                   |    862.549|    97.342|4M         |        149588|          156|
|ppo_lstm|CartPoleNoVel-v1               |    500.000|     0.000|100k       |        150000|          300|
|ppo_lstm|MountainCarContinuousNoVel-v0  |     91.469|     1.776|300k       |        149882|         1340|
|ppo_lstm|PendulumNoVel-v1               |   -217.933|   140.094|100k       |        150000|          750|
|qrdqn   |Acrobot-v1                     |    -69.135|     9.967|100k       |        149949|         2138|
|qrdqn   |AsteroidsNoFrameskip-v4        |   2185.303|  1097.172|10M        |        599784|           66|
|qrdqn   |BeamRiderNoFrameskip-v4        |  17122.941| 10769.997|10M        |        596483|           17|
|qrdqn   |BreakoutNoFrameskip-v4         |    393.600|    79.828|10M        |        579711|           40|
|qrdqn   |CartPole-v1                    |    500.000|     0.000|50k        |        150000|          300|
|qrdqn   |EnduroNoFrameskip-v4           |   3231.200|  1311.801|10M        |        585728|            5|
|qrdqn   |LunarLander-v2                 |     70.236|   225.491|100k       |        149957|          522|
|qrdqn   |MountainCar-v0                 |   -106.042|    15.536|120k       |        149943|         1414|
|qrdqn   |MsPacmanNoFrameskip-v4         |    997.867|   877.130|10M        |        604914|          225|
|qrdqn   |PongNoFrameskip-v4             |     20.492|     0.687|10M        |        597443|           63|
|qrdqn   |QbertNoFrameskip-v4            |  14799.728|  2917.629|10M        |        600773|           92|
|qrdqn   |RoadRunnerNoFrameskip-v4       |  42325.424|  8361.161|10M        |        591016|           59|
|qrdqn   |SeaquestNoFrameskip-v4         |   2557.576|    76.951|10M        |        596275|           66|
|qrdqn   |SpaceInvadersNoFrameskip-v4    |   1899.928|   823.488|10M        |        597218|           69|
|sac     |Ant-v3                         |   4615.791|  1354.111|1M         |        149074|          165|
|sac     |AntBulletEnv-v0                |   3073.114|   175.148|1M         |        150000|          150|
|sac     |BipedalWalker-v3               |    297.668|    33.060|500k       |        149530|          136|
|sac     |BipedalWalkerHardcore-v3       |      4.423|   103.910|10M        |        149794|           88|
|sac     |HalfCheetah-v3                 |   9535.451|   100.470|1M         |        150000|          150|
|sac     |HalfCheetahBulletEnv-v0        |   2792.170|    12.088|1M         |        150000|          150|
|sac     |Hopper-v3                      |   2325.547|  1129.676|1M         |        149841|          236|
|sac     |HopperBulletEnv-v0             |   2603.494|   164.322|1M         |        149724|          151|
|sac     |Humanoid-v3                    |   6232.287|   279.885|2M         |        149460|          150|
|sac     |LunarLanderContinuous-v2       |    260.390|    65.467|500k       |        149634|          672|
|sac     |MountainCarContinuous-v0       |     94.679|     1.134|50k        |        149966|         1443|
|sac     |Pendulum-v1                    |   -156.995|    88.714|20k        |        150000|          750|
|sac     |ReacherBulletEnv-v0            |     18.062|     9.729|300k       |        150000|         1000|
|sac     |Swimmer-v3                     |    345.568|     3.084|1M         |        150000|          150|
|sac     |Walker2DBulletEnv-v0           |   2292.266|    13.970|1M         |        149983|          150|
|sac     |Walker2d-v3                    |   3863.203|   254.347|1M         |        149309|          150|
|td3     |Ant-v3                         |   5813.274|   589.773|1M         |        149393|          151|
|td3     |AntBulletEnv-v0                |   3300.026|    54.640|1M         |        150000|          150|
|td3     |BipedalWalker-v3               |    305.990|    56.886|1M         |        149999|          224|
|td3     |BipedalWalkerHardcore-v3       |    -98.116|    16.087|10M        |        150000|           75|
|td3     |HalfCheetah-v3                 |   9655.666|   969.916|1M         |        150000|          150|
|td3     |HalfCheetahBulletEnv-v0        |   2821.641|    19.722|1M         |        150000|          150|
|td3     |Hopper-v3                      |   3606.390|     4.027|1M         |        150000|          150|
|td3     |HopperBulletEnv-v0             |   2681.609|    27.806|1M         |        149486|          150|
|td3     |Humanoid-v3                    |   5566.687|    14.544|2M         |        150000|          150|
|td3     |LunarLanderContinuous-v2       |    207.451|    67.562|300k       |        149488|          337|
|td3     |MountainCarContinuous-v0       |     93.483|     0.075|300k       |        149976|         2275|
|td3     |Pendulum-v1                    |   -151.855|    90.227|20k        |        150000|          750|
|td3     |ReacherBulletEnv-v0            |     17.114|     9.750|300k       |        150000|         1000|
|td3     |Swimmer-v3                     |    359.127|     1.244|1M         |        150000|          150|
|td3     |Walker2DBulletEnv-v0           |   2213.672|   230.558|1M         |        149800|          152|
|td3     |Walker2d-v3                    |   4717.823|    46.303|1M         |        150000|          150|
|tqc     |Ant-v3                         |   3339.362|  1969.906|1M         |        149583|          202|
|tqc     |AntBulletEnv-v0                |   3456.717|   248.733|1M         |        150000|          150|
|tqc     |BipedalWalker-v3               |    329.808|    45.083|500k       |        149682|          254|
|tqc     |BipedalWalkerHardcore-v3       |    235.226|   110.569|2M         |        149032|          131|
|tqc     |FetchPickAndPlace-v1           |     -9.331|     6.850|1M         |        150000|         3000|
|tqc     |FetchPush-v1                   |     -8.799|     5.438|1M         |        150000|         3000|
|tqc     |FetchReach-v1                  |     -1.659|     0.873|20k        |        150000|         3000|
|tqc     |FetchSlide-v1                  |    -29.210|    11.387|3M         |        150000|         3000|
|tqc     |HalfCheetah-v3                 |  12089.939|   127.440|1M         |        150000|          150|
|tqc     |HalfCheetahBulletEnv-v0        |   3675.299|    17.681|1M         |        150000|          150|
|tqc     |Hopper-v3                      |   3754.199|     8.276|1M         |        150000|          150|
|tqc     |HopperBulletEnv-v0             |   2662.373|   206.210|1M         |        149881|          151|
|tqc     |Humanoid-v3                    |   7239.320|  1647.498|2M         |        149508|          165|
|tqc     |LunarLanderContinuous-v2       |    277.956|    25.466|500k       |        149928|          706|
|tqc     |MountainCarContinuous-v0       |     63.641|    45.259|50k        |        149796|          186|
|tqc     |PandaPickAndPlace-v1           |     -8.024|     6.674|1M         |        150000|         3000|
|tqc     |PandaPush-v1                   |     -6.405|     6.400|1M         |        150000|         3000|
|tqc     |PandaReach-v1                  |     -1.768|     0.858|20k        |        150000|         3000|
|tqc     |PandaSlide-v1                  |    -27.497|     9.868|3M         |        150000|         3000|
|tqc     |PandaStack-v1                  |    -96.915|    17.240|1M         |        150000|         1500|
|tqc     |Pendulum-v1                    |   -151.340|    87.893|20k        |        150000|          750|
|tqc     |ReacherBulletEnv-v0            |     18.255|     9.543|300k       |        150000|         1000|
|tqc     |Swimmer-v3                     |    339.423|     1.486|1M         |        150000|          150|
|tqc     |Walker2DBulletEnv-v0           |   2508.934|   614.624|1M         |        149572|          159|
|tqc     |Walker2d-v3                    |   4380.720|   500.489|1M         |        149606|          152|
|tqc     |parking-v0                     |     -6.762|     2.690|100k       |        149983|         7528|
|trpo    |Acrobot-v1                     |    -83.114|    18.648|100k       |        149976|         1783|
|trpo    |Ant-v3                         |   4982.301|   663.761|1M         |        149909|          153|
|trpo    |AntBulletEnv-v0                |   2560.621|    52.064|2M         |        150000|          150|
|trpo    |BipedalWalker-v3               |    182.339|   145.570|1M         |        148440|          148|
|trpo    |CartPole-v1                    |    500.000|     0.000|100k       |        150000|          300|
|trpo    |HalfCheetah-v3                 |   1785.476|    68.672|1M         |        150000|          150|
|trpo    |HalfCheetahBulletEnv-v0        |   2758.752|   327.032|2M         |        150000|          150|
|trpo    |Hopper-v3                      |   3618.386|   356.768|1M         |        149575|          152|
|trpo    |HopperBulletEnv-v0             |   2565.416|   410.298|1M         |        149640|          154|
|trpo    |LunarLander-v2                 |    133.166|   112.173|200k       |        149088|          230|
|trpo    |LunarLanderContinuous-v2       |    262.387|    21.428|200k       |        149925|          501|
|trpo    |MountainCar-v0                 |   -107.278|    13.231|100k       |        149974|         1398|
|trpo    |MountainCarContinuous-v0       |     92.489|     0.355|50k        |        149971|         1732|
|trpo    |Pendulum-v1                    |   -174.631|   127.577|100k       |        150000|          750|
|trpo    |ReacherBulletEnv-v0            |     14.741|    11.559|300k       |        150000|         1000|
|trpo    |Swimmer-v3                     |    365.663|     2.087|1M         |        150000|          150|
|trpo    |Walker2DBulletEnv-v0           |   1483.467|   823.468|2M         |        149860|          197|
|trpo    |Walker2d-v3                    |   4933.148|  1452.538|1M         |        149054|          163|
