import argparse
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecVideoRecorder

from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
parser.add_argument("-o", "--output-folder", help="Output folder", type=str, default="logs/videos/")
parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument("--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)")
parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
args = parser.parse_args()

env_id = args.env
algo = args.algo
folder = args.folder
video_folder = args.output_folder
seed = args.seed
deterministic = args.deterministic
video_length = args.n_timesteps
n_envs = args.n_envs

if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    print(f"Loading latest experiment, id={args.exp_id}")
# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
else:
    log_path = os.path.join(folder, algo)

model_path = os.path.join(log_path, f"{env_id}.zip")

stats_path = os.path.join(log_path, env_id)
hyperparams, stats_path = get_saved_hyperparams(stats_path)


is_atari = "NoFrameskip" in env_id

env = create_test_env(
    env_id,
    n_envs=n_envs,
    stats_path=stats_path,
    seed=seed,
    log_dir=None,
    should_render=not args.no_render,
    hyperparams=hyperparams,
)

model = ALGOS[algo].load(model_path)

obs = env.reset()

# Note: apparently it renders by default
env = VecVideoRecorder(
    env,
    video_folder,
    record_video_trigger=lambda x: x == 0,
    video_length=video_length,
    name_prefix=f"{algo}-{env_id}",
)

env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=deterministic)
    obs, _, _, _ = env.step(action)

# Workaround for https://github.com/openai/gym/issues/893
if n_envs == 1 and "Bullet" not in env_id and not is_atari:
    env = env.venv
    # DummyVecEnv
    while isinstance(env, VecEnvWrapper):
        env = env.venv
    if isinstance(env, DummyVecEnv):
        env.envs[0].env.close()
    else:
        env.close()
else:
    # SubprocVecEnv
    env.close()
