import os
import re
import shutil

import torch as th
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy


class CppExporter(object):
    def __init__(self, model: BaseAlgorithm, directory: str, name: str):
        """
        C++ module exporter

        :param BaseAlgorithm model: The algorithm that should be exported
        :param str directory: Output directory
        :param str name: The module name
        """
        self.model = model
        self.directory = directory
        self.name = name.replace("-", "_")

        # Templates directory is found relatively to this script (../cpp)
        self.template_directory = "/".join(__file__.split("/")[:-1] + ["..", "cpp"])
        self.vars = {}

        # Name of the asset (.pt file)
        self.asset_fnames = []
        self.cpp_fname = None

    def generate_directory(self):
        """
        Generates the target directory if it doesn't exists
        """

        def ignore(directory, files):
            if directory == self.template_directory:
                return [".gitignore", "model_template.h", "model_template.cpp"]

            return []

        if not os.path.isdir(self.directory):
            shutil.copytree(self.template_directory, self.directory, ignore=ignore)

    def generate_observation_preprocessing(self):
        """
        Generates observation preprocessing code for this model

        :raises NotImplementedError: If the observation space is not supported
        """
        observation_space = self.model.env.observation_space
        policy = self.model.policy
        preprocess_observation = ""
        self.vars["OBSERVATION_SPACE"] = repr(observation_space)

        if isinstance(observation_space, spaces.Box):
            if is_image_space(observation_space) and policy.normalize_images:
                # Normalizing image pixels
                preprocess_observation += "result = observation / 255.;\n"
            else:
                # Keeping observation as it is
                preprocess_observation += "result = observation;\n"
        elif isinstance(observation_space, spaces.Discrete):
            # Applying one hot representation
            preprocess_observation += f"result = torch::one_hot(observation, {observation_space.n});\n"
        elif isinstance(observation_space, spaces.MultiDiscrete):
            # Applying multiple one hot representation (using C++ function)
            classes = ",".join(map(str, observation_space.nvec))
            preprocess_observation += "torch::Tensor classes = torch::tensor({%s});\n" % classes
            preprocess_observation += f"result = multi_one_hot(observation, classes);\n"
        else:
            raise NotImplementedError(f"C++ exporting does not support observation {observation_space}")

        self.vars["PREPROCESS_OBSERVATION"] = preprocess_observation

    def generate_action_processing(self):
        """
        Generates the action post-processing

        :raises NotImplementedError: If the action space is not supported
        """
        action_space = self.model.env.action_space
        process_action = ""
        self.vars["ACTION_SPACE"] = repr(action_space)

        if isinstance(action_space, spaces.Box):
            # Handling action clipping
            low_values = ",".join([f"(float){x}" for x in action_space.low])
            process_action += "torch::Tensor action_low = torch::tensor({%s});\n" % low_values
            high_values = ",".join([f"(float){x}" for x in action_space.high])
            process_action += "torch::Tensor action_high = torch::tensor({%s});\n" % high_values

            if self.model.policy.squash_output:
                # Unscaling the action assuming it lies in [-1, 1], since squash networks use Tanh as
                # final activation functions
                process_action += "result = action_low + (0.5 * (action + 1.0) * (action_high - action_low));\n"
            else:
                # Clipping not squashed action
                process_action += "result = torch::clip(action, action_low, action_high);\n"
        elif isinstance(action_space, spaces.Discrete) or isinstance(action_space, spaces.MultiDiscrete):
            # Keeping input as it is
            process_action += "result = action;\n"
        else:
            raise NotImplementedError(f"C++ exporting does not support processing action {action_space}")

        self.vars["PROCESS_ACTION"] = process_action

    def export_code(self):
        """
        Export the C++ code
        """
        self.vars["CLASS_NAME"] = self.name
        fname = self.name.lower()

        self.vars["FILE_NAME"] = fname
        self.cpp_fname = f"src/baselines3_models/{fname}.cpp"
        target_header = self.directory + f"/include/baselines3_models/{fname}.h"
        target_cpp = self.directory + "/" + self.cpp_fname

        self.generate_observation_preprocessing()
        self.generate_action_processing()

        self.render("model_template.h", target_header)
        self.render("model_template.cpp", target_cpp)

    def render(self, template: str, target: str):
        """
        Renders some template, replacing self.vars variables by their values

        :param str template: The template name
        :param str target: The target file
        """
        with open(self.template_directory + "/" + template, "r") as template_f:
            with open(target, "w") as target_f:
                data = template_f.read()
                for var in self.vars:
                    data = data.replace(var, self.vars[var])
                target_f.write(data)

        print("Generated " + target)

    def update_cmake(self):
        """
        Updates the target's CMakeLists.txt, adding files in static and sources section

        :raises ValueError: If a section can't be found in the CMakeLists
        """
        cmake_contents = open(self.directory + "/CMakeLists.txt", "r").read()

        def add_to_section(section_name: str, fname: str, contents: str):
            pattern = f"#{section_name}(.+)#!{section_name}"
            flags = re.MULTILINE + re.DOTALL

            match = re.search(pattern, cmake_contents, flags=flags)

            if match is None:
                raise ValueError(f"Couldn't find {section_name} section in CMakeLists.txt")

            files = match[1].strip()
            if files:
                files = list(map(str.strip, files.split("\n")))
            else:
                files = []

            if fname not in files:
                print(f"Adding {fname} to CMake {section_name}")
                files.append(fname)

            new_section = f"#{section_name}\n" + ("\n".join(files)) + "\n" + f"#!{section_name}"

            return re.sub(pattern, new_section, contents, flags=flags)

        for asset in self.asset_fnames:
            cmake_contents = add_to_section("static", asset, cmake_contents)
        cmake_contents = add_to_section("sources", self.cpp_fname, cmake_contents)

        with open(self.directory + "/CMakeLists.txt", "w") as f:
            f.write(cmake_contents)

    def export_model(self):
        """
        Export the Algorithm's model using Pytorch's JIT script tracer

        :raises NotImplementedError: If the policy is not supported
        """
        policy = self.model.policy
        obs = th.Tensor(self.model.env.reset())

        def get_fname(suffix):
            asset_fname = f"assets/{self.name.lower()}_{suffix}.pt"
            fname = self.directory + "/" + asset_fname
            return asset_fname, fname

        traced = {
            "actor": None,
            "q": None,
            "v": None,
        }

        obs = preprocess_obs(obs, self.model.env.observation_space)

        if isinstance(policy, TD3Policy):
            # Actor extract features and apply mu
            actor_model = th.nn.Sequential(policy.actor.features_extractor, policy.actor.mu)
            traced["actor"] = th.jit.trace(actor_model, obs)

            # Value function is a combination of actor and Q
            class TD3PolicyValue(th.nn.Module):
                def __init__(self, policy : TD3Policy, actor_model: th.nn.Module):
                    super(TD3PolicyValue, self).__init__()

                    self.actor = actor_model
                    self.critic = policy.critic

                def forward(self, obs):
                    action = self.actor_model(obs)
                    critic_features = self.critic.features_extractor(obs)
                    return self.critic.q_networks[0](th.cat([critic_features, action], dim=1))

            action = policy.actor.mu(policy.actor.extract_features(obs))
            v_model = TD3PolicyValue(policy, actor_model)
            traced["v"] = th.jit.trace(v_model, obs)
            self.vars["POLICY_TYPE"] = "ACTOR_VALUE"
        elif isinstance(policy, SACPolicy):
            # Feature extractor, latent pi and mu
            if self.model.use_sde:
                # XXX: Check for bijector ?
                actor_model = th.nn.Sequential(policy.actor.features_extractor, policy.actor.latent_pi, policy.actor.mu)
            else:
                actor_model = th.nn.Sequential(policy.actor.features_extractor,
                                               policy.actor.latent_pi, policy.actor.mu, th.nn.Tanh())
            traced["actor"] = th.jit.trace(actor_model, obs)

            class SACPolicyValue(th.nn.Module):
                def __init__(self, policy : SACPolicy, actor_model: th.nn.Module):
                    super(SACPolicyValue, self).__init__()

                    self.actor_model = actor_model
                    self.critic = policy.critic

                def forward(self, obs):
                    action = self.actor_model(obs)
                    critic_features = self.critic.features_extractor(obs)
                    return self.critic.q_networks[0](th.cat([critic_features, action], dim=1))

            v_model = SACPolicyValue(policy, actor_model)
            traced["v"] = th.jit.trace(v_model, obs)
            self.vars["POLICY_TYPE"] = "ACTOR_VALUE"
        elif isinstance(policy, ActorCriticPolicy):
            # Actor is feature extractor, mpl and action net
            actor_model = th.nn.Sequential(policy.features_extractor, policy.mlp_extractor.policy_net, policy.action_net)
            traced["actor"] = th.jit.trace(actor_model, obs)

            # The value network is computed directly in ActorCriticPolicy (and not the Q network)
            value_model = th.nn.Sequential(policy.features_extractor, policy.mlp_extractor.value_net, policy.value_net)
            traced["v"] = th.jit.trace(value_model, obs)

            if isinstance(self.model.env.action_space, spaces.Discrete):
                self.vars["POLICY_TYPE"] = "ACTOR_VALUE_DISCRETE"
            else:
                self.vars["POLICY_TYPE"] = "ACTOR_VALUE"
        elif isinstance(policy, DQNPolicy):
            # For DQN, we only use one Q network that outputs Q(s,a) for all possible actions, it is then
            # both used for action prediction using argmax and for value prediction
            q_model = th.nn.Sequential(policy.q_net.features_extractor, policy.q_net.q_net)
            traced["q"] = th.jit.trace(q_model, obs)
            self.vars["POLICY_TYPE"] = "QNET_ALL"
        else:
            raise NotImplementedError(f"C++ exporting does not support policy {policy}")

        for entry in traced.keys():
            var = f"MODEL_{entry.upper()}"
            if traced[entry] is None:
                self.vars[var] = ""
            else:
                asset_fname, fname = get_fname(entry)
                traced[entry].save(fname)
                print(f"Generated {fname}")
                self.asset_fnames.append(asset_fname)
                self.vars[var] = asset_fname

    def export(self):
        self.generate_directory()
        self.export_model()
        self.export_code()
        self.update_cmake()
