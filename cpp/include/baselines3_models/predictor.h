#pragma once

#include <string>
#include <torch/script.h>

namespace baselines3_models {
class Predictor {
public:
  enum PolicyType {
    // If we have an actor network and a value network
    ACTOR_VALUE,
    ACTOR_VALUE_DISCRETE,
    // The network is a Q-Network: outputs Q(s,a) for all a for a given s
    QNET_ALL
  };

  Predictor(std::string actor_filename, std::string q_filename, std::string v_filename);

  torch::Tensor predict(torch::Tensor &observation, bool unscale_action = true);
  double value(torch::Tensor &observation);

  std::vector<float> predict_vector(std::vector<float> obs);

  virtual torch::Tensor preprocess_observation(torch::Tensor &observation);
  virtual torch::Tensor process_action(torch::Tensor &action);
  virtual std::vector<torch::Tensor> enumerate_actions();

protected:
  torch::jit::script::Module model_actor;
  torch::jit::script::Module model_q;
  torch::jit::script::Module model_v;
  PolicyType policy_type;
};
} // namespace baselines3_models