import torch as th
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.dqn.policies import DQNPolicy


class CppExporter(object):
    def __init__(self, model: BaseAlgorithm, directory: str, name: str):
        self.model = model
        self.directory = directory
        self.name = name.replace("-", "_")
        self.template_directory = dir = "/".join(__file__.split("/")[:-1] + ["..", "cpp"])
        self.vars = {}

    def generate_observation_preprocessing(self):
        observation_space = self.model.env.observation_space
        policy = self.model.policy
        preprocess_observation = ""
        self.vars["OBSERVATION_SPACE"] = repr(observation_space)

        if isinstance(observation_space, spaces.Box):
            if is_image_space(observation_space) and policy.normalize_images:
                preprocess_observation += "result = observation / 255.;\n"
            else:
                preprocess_observation += "result = observation;\n"
        elif isinstance(observation_space, spaces.Discrete):
            preprocess_observation += f"result = torch::one_hot(observation, {observation_space.n});\n"
        elif isinstance(observation_space, spaces.MultiDiscrete):
            classes = ",".join(map(str, observation_space.nvec))
            preprocess_observation += "torch::Tensor classes = torch::tensor({%s});\n" % classes
            preprocess_observation += f"result = multi_one_hot(observation, classes);\n"
        else:
            raise NotImplementedError(f"C++ exporting does not support observation {observation_space}")

        self.vars["PREPROCESS_OBSERVATION"] = preprocess_observation

    def generate_action_processing(self):
        action_space = self.model.env.action_space
        process_action = ""
        self.vars["ACTION_SPACE"] = repr(action_space)

        if isinstance(action_space, spaces.Box):
            if self.model.policy.squash_output:
                low_values = ",".join([f"(float){x}" for x in action_space.low])
                process_action += "torch::Tensor action_low = torch::tensor({%s});\n" % low_values
                high_values = ",".join([f"(float){x}" for x in action_space.high])
                process_action += "torch::Tensor action_high = torch::tensor({%s});\n" % high_values

                process_action += "result = action_low + (0.5 * (action + 1.0) * (action_high - action_low));\n"
            else:
                process_action += "result = action;\n"
        if isinstance(action_space, spaces.Discrete):
            process_action += "result = action;\n"
        elif isinstance(action_space, spaces.Box):
            process_action += "result = action;\n"
        else:
            raise NotImplementedError(f"C++ exporting does not support processing action {action_space}")

        self.vars["PROCESS_ACTION"] = process_action

    def export_code(self):
        self.vars["CLASS_NAME"] = self.name
        fname = self.name.lower()
        self.vars["FILE_NAME"] = fname
        target_header = self.directory + f"/include/baselines3_models/{fname}.h"
        target_cpp = self.directory + f"/src/baselines3_models/{fname}.cpp"

        self.generate_observation_preprocessing()
        self.generate_action_processing()

        self.render("model_template.h", target_header)
        self.render("model_template.cpp", target_cpp)

    def render(self, template: str, target: str):
        with open(self.template_directory + "/" + template, "r") as template_f:
            with open(target, "w") as target_f:
                data = template_f.read()
                for var in self.vars:
                    data = data.replace(var, self.vars[var])
                target_f.write(data)

        print("Generated " + target)

    def export_model(self):
        policy = self.model.policy
        obs = th.Tensor(self.model.env.reset())
        asset_fname = f"assets/{self.name}_model.pt"
        fname = self.directory + "/" + asset_fname
        traced_script_module = None

        if isinstance(policy, TD3Policy):
            traced_script_module = th.jit.trace(policy.actor.mu, policy.actor.extract_features(obs))
            self.vars["POLICY_TYPE"] = "ACTOR_MU"
        elif isinstance(policy, SACPolicy):
            model = th.nn.Sequential(policy.actor.latent_pi, policy.actor.mu)
            traced_script_module = th.jit.trace(model, policy.actor.extract_features(obs))
            self.vars["POLICY_TYPE"] = "ACTOR_MU"
        elif isinstance(policy, ActorCriticPolicy):
            model = th.nn.Sequential(policy.mlp_extractor.policy_net, policy.action_net)
            traced_script_module = th.jit.trace(model, policy.extract_features(obs))
            self.vars["POLICY_TYPE"] = "ACTOR_MU"
        elif isinstance(policy, DQNPolicy):
            traced_script_module = th.jit.trace(policy.q_net.q_net, policy.q_net.extract_features(obs))
            self.vars["POLICY_TYPE"] = "QNET_SCAN"
        else:
            raise NotImplementedError(f"C++ exporting does not support policy {policy}")

        if traced_script_module is not None:
            print(f"Generated {fname}")
            traced_script_module.save(fname)
            self.vars["MODEL_FNAME"] = asset_fname

    def export(self):
        self.export_model()
        self.export_code()
