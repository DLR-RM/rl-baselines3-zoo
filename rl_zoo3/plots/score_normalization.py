"""
Min and Max score for each env for normalization when plotting.
Min score corresponds to random agent.
Max score corresponds to acceptable performance, for instance
human level performance in the case of Atari games.
"""
from typing import NamedTuple

import numpy as np


class ReferenceScore(NamedTuple):
    env_id: str
    min: float
    max: float


reference_scores = [
    # PyBullet Envs
    ReferenceScore("HalfCheetahBulletEnv-v0", -1400, 3000),
    ReferenceScore("AntBulletEnv-v0", 300, 3500),
    ReferenceScore("HopperBulletEnv-v0", 20, 2500),
    ReferenceScore("Walker2DBulletEnv-v0", 200, 2500),
]

# Alternative scaling
# Min is a poorly optimized algorithm
# reference_scores = [
#     ReferenceScore("HalfCheetahBulletEnv-v0", 1000, 3000),
#     ReferenceScore("AntBulletEnv-v0", 1000, 3500),
#     ReferenceScore("HopperBulletEnv-v0", 1000, 2500),
#     ReferenceScore("Walker2DBulletEnv-v0", 500, 2500),
# ]

min_max_score_per_env = {reference_score.env_id: reference_score for reference_score in reference_scores}


def normalize_score(score: np.ndarray, env_id: str) -> np.ndarray:
    """
    Normalize score to be in [0, 1] where 1 is maximal performance.

    :param score: unnormalized score
    :param env_id: environment id
    :return: normalized score
    """
    if env_id not in min_max_score_per_env:
        raise KeyError(f"No reference score for {env_id}")
    reference_score = min_max_score_per_env[env_id]
    return (score - reference_score.min) / (reference_score.max - reference_score.min)


# From rliable, for atari games:
#
# RANDOM_SCORES = {
#  'Alien': 227.8,
#  'Amidar': 5.8,
#  'Assault': 222.4,
#  'Asterix': 210.0,
#  'BankHeist': 14.2,
#  'BattleZone': 2360.0,
#  'Boxing': 0.1,
#  'Breakout': 1.7,
#  'ChopperCommand': 811.0,
#  'CrazyClimber': 10780.5,
#  'DemonAttack': 152.1,
#  'Freeway': 0.0,
#  'Frostbite': 65.2,
#  'Gopher': 257.6,
#  'Hero': 1027.0,
#  'Jamesbond': 29.0,
#  'Kangaroo': 52.0,
#  'Krull': 1598.0,
#  'KungFuMaster': 258.5,
#  'MsPacman': 307.3,
#  'Pong': -20.7,
#  'PrivateEye': 24.9,
#  'Qbert': 163.9,
#  'RoadRunner': 11.5,
#  'Seaquest': 68.4,
#  'UpNDown': 533.4
# }
#
# HUMAN_SCORES = {
#  'Alien': 7127.7,
#  'Amidar': 1719.5,
#  'Assault': 742.0,
#  'Asterix': 8503.3,
#  'BankHeist': 753.1,
#  'BattleZone': 37187.5,
#  'Boxing': 12.1,
#  'Breakout': 30.5,
#  'ChopperCommand': 7387.8,
#  'CrazyClimber': 35829.4,
#  'DemonAttack': 1971.0,
#  'Freeway': 29.6,
#  'Frostbite': 4334.7,
#  'Gopher': 2412.5,
#  'Hero': 30826.4,
#  'Jamesbond': 302.8,
#  'Kangaroo': 3035.0,
#  'Krull': 2665.5,
#  'KungFuMaster': 22736.3,
#  'MsPacman': 6951.6,
#  'Pong': 14.6,
#  'PrivateEye': 69571.3,
#  'Qbert': 13455.0,
#  'RoadRunner': 7845.0,
#  'Seaquest': 42054.7,
#  'UpNDown': 11693.2
# }
