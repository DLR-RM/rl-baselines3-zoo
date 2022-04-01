#pragma once

#include <string>
#include <torch/script.h>

namespace baselines3_models {
class Predictor {
public:
  enum PolicyType {
    // The module is an actor's µ: directly outputs action from state
    ACTOR_MU,
    // The network is a Q-Network: outputs Q(s,a) for all a for a given s
    QNET_ALL
  };

  Predictor(std::string model_filename);

  torch::Tensor predict(torch::Tensor &observation);

  std::vector<float> predict_vector(std::vector<float> obs);

  virtual torch::Tensor preprocess_observation(torch::Tensor &observation);
  virtual torch::Tensor process_action(torch::Tensor &action);
  virtual std::vector<torch::Tensor> enumerate_actions();

protected:
  torch::jit::script::Module module;
  PolicyType policy_type;
};
} // namespace baselines3_models