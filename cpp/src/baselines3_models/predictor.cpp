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
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(processed_observation);

  if (policy_type == ACTOR_MU) {
    action = module.forward(inputs).toTensor();
    action = process_action(action);
  } else if (policy_type == QNET_ALL) {
    auto q_values = module.forward(inputs).toTensor();
    action = torch::argmax(q_values);
  } else {
    throw std::runtime_error("Unknown policy type");
  }

  return action;
}

std::vector<float> Predictor::predict_vector(std::vector<float> obs) {
  torch::Tensor observation = torch::from_blob(obs.data(), obs.size());
  torch::Tensor action = predict(observation);
  action = action.contiguous().to(torch::kFloat32);
  std::vector<float> result(action.data_ptr<float>(),
                            action.data_ptr<float>() + action.numel());
  return result;
}

torch::Tensor Predictor::preprocess_observation(torch::Tensor &observation) {
  return observation;
}

torch::Tensor Predictor::process_action(torch::Tensor &action) {
  return action;
}

std::vector<torch::Tensor> Predictor::enumerate_actions() {
  std::vector<torch::Tensor> result;
  return result;
}

} // namespace baselines3_models