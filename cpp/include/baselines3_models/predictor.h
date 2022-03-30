#pragma once

#include <string>
#include <torch/script.h>

namespace baselines3_models {
class Predictor {
public:
  enum PolicyType {
    ACTOR_MU
  };

  Predictor(std::string model_filename);

  torch::Tensor predict(torch::Tensor &observation);

  virtual torch::Tensor preprocess_observation(torch::Tensor &observation);
  virtual torch::Tensor process_action(torch::Tensor &action);

protected:
  torch::jit::script::Module module;
  PolicyType policy_type;
};
} // namespace baselines3_models