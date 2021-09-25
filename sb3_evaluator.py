import os
from chrono import Chrono
import argparse
import numpy as np
from itertools import count
from utils import ALGOS, create_test_env, get_saved_hyperparams


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    args = parser.parse_args()
    return args


def read_name(filename):
    """
    :param filename: the file name, including the path
    :return: fields
    """
    fields = filename.split('#')
    tmp = fields[0]
    env_name = tmp.split('/')
    env_name = env_name[-1]
    algo = fields[1]
    team_name = fields[2]
    name = team_name.split('.')
    return env_name, name[0], algo


def get_scores(args, folder, policy_file, env_name, algo, stats_path, hyperparams):
    """
    """
    env = create_test_env(
        env_name,
        n_envs=1,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=folder + "/../Logs",
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs={},
    )
    fields = policy_file.split('.')
    model = ALGOS[algo].load(folder + "/" + fields[0], env)
    policy = model.policy
    scores = evaluate_pol(env, policy, False)
    return scores


def evaluate_pol(env, policy, deterministic):
    """
    Function to evaluate a policy over 900 episodes
    :param env: the evaluation environment
    :param policy: the evaluated policy
    :param deterministic: whether the evaluation uses a deterministic policy
    :return: the obtained vector of 900 scores
    """
    scores = []
    state = None
    for i in range(100):
        osb = env.reset()
        total_reward = 0

        for _ in count():
            action, state = policy.predict(osb, state=state, deterministic=deterministic)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            osb = next_state

            if done:
                scores.append(total_reward)
                break
    scores = np.array(scores)
    return scores


class Evaluator:
    """
    A class to evaluate a set of policies stored into the same folder and ranking them according to their scores
    """
    def __init__(self):
        self.env_dic = {}
        self.score_dic = {}

    def load_policies(self, folder) -> None:
        """
         :param: folder : name of the folder containing policies
         Output : none (policies of the folder stored in self.env_dict)        
         """
        listdir = os.listdir(folder)
        args = parse()
        stats_path = folder
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

        for policy_file in listdir:
            print(policy_file)
            env_name, team_name, algo = read_name(policy_file)

            if env_name in self.env_dic:
                scores = get_scores(args, folder, policy_file, env_name, algo, stats_path, hyperparams)
                self.score_dic[env_name][team_name] = [scores.mean(), scores.std()]
            else:
                scores = get_scores(args, folder, policy_file, env_name, algo, stats_path, hyperparams)
                tmp_dic = {team_name: [scores.mean(), scores.std()]}
                self.score_dic[env_name] = tmp_dic

    def display_hall_of_fame(self) -> None:
        """
        Display the hall of fame of all the evaluated policies
        :return: nothing
        """
        print("Hall of fame")
        for env, dico in self.score_dic.items():
            print("Environment :", env)
            for team, score in sorted(dico.items()):
                print("team: ", team, "mean: ", score[0], "std: ", score[1])


if __name__ == '__main__':
    directory = os.getcwd() + '/data/policies/'
    ev = Evaluator()
    c = Chrono()
    ev.load_policies(directory)
    ev.display_hall_of_fame()
    c.stop()
