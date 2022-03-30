#include "baselines3_models/predictor.h"
#include "cmrc/cmrc.hpp"

CMRC_DECLARE(baselines3_model);

namespace baselines3_models {
Predictor::Predictor(std::string model_filename) {
  auto fs = cmrc::baselines3_model::get_filesystem();
  auto f = fs.open(model_filename);
  std::string data(f.begin(), f.end());
  std::istringstream stream(data);
  module = torch::jit::load(stream);
}

torch::Tensor Predictor::predict(torch::Tensor &observation) {
  c10::InferenceMode guard;
  torch::Tensor processed_observation = preprocess_observation(observation);
  at::Tensor action;

  if (policy_type == ACTOR_MU) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(processed_observation);

    action = module.forward(inputs).toTensor();
  } else {
    throw std::runtime_error("Unknown policy type");
  }

  return process_action(action);
}

torch::Tensor Predictor::preprocess_observation(torch::Tensor &observation) {
  return observation;
}

torch::Tensor Predictor::process_action(torch::Tensor &action) {
  return action;
}

} // namespace baselines3_models