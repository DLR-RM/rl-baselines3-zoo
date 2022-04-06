#include "baselines3_models/predictor.h"
#include "cmrc/cmrc.hpp"

CMRC_DECLARE(baselines3_model);

namespace baselines3_models {

static void _load_model(std::string filename,
                        torch::jit::script::Module &model) {
  if (filename != "") {
    auto fs = cmrc::baselines3_model::get_filesystem();
    auto f = fs.open(filename);
    std::string data(f.begin(), f.end());
    std::istringstream stream(data);
    model = torch::jit::load(stream);
  }
}

Predictor::Predictor(std::string actor_filename, std::string q_filename,
                     std::string v_filename) {

  _load_model(actor_filename, model_actor);
  _load_model(q_filename, model_q);
  _load_model(v_filename, model_v);
}

torch::Tensor Predictor::predict(torch::Tensor &observation,
                                 bool unscale_action) {
  c10::InferenceMode guard;
  torch::Tensor processed_observation = preprocess_observation(observation);
  at::Tensor action;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(processed_observation.unsqueeze(0));

  if (policy_type == ACTOR_Q || policy_type == ACTOR_VALUE || policy_type == ACTOR_VALUE_DISCRETE) {
    action = model_actor.forward(inputs).toTensor();
    if (unscale_action) {
      action = process_action(action);
    }
    if (policy_type == ACTOR_VALUE_DISCRETE) {
      action = torch::argmax(action);
    }
  } else if (policy_type == QNET_ALL) {
    auto q_values = model_q.forward(inputs).toTensor();
    action = torch::argmax(q_values);
  } else {
    throw std::runtime_error("Unknown policy type");
  }

  return action;
}

double Predictor::value(torch::Tensor &observation) {
  double value = 0.0;

  torch::Tensor processed_observation = preprocess_observation(observation);
  at::Tensor action;
  std::vector<torch::jit::IValue> inputs;

  if (policy_type == ACTOR_Q) {
    auto action = predict(observation, false);
    std::vector<torch::Tensor> tensor_vec{ processed_observation, action };
    inputs.push_back(torch::cat({ tensor_vec }).unsqueeze(0));

    auto q = model_q.forward(inputs).toTensor();
    value = q.data_ptr<float>()[0];
  } else if (policy_type == ACTOR_VALUE || policy_type == ACTOR_VALUE_DISCRETE) {
    inputs.push_back(processed_observation.unsqueeze(0));
    auto v = model_v.forward(inputs).toTensor();
    value = v.data_ptr<float>()[0];
  } else if (policy_type == QNET_ALL) {
    inputs.push_back(processed_observation.unsqueeze(0));
    auto q = model_q.forward(inputs).toTensor();
    value = torch::max(q).data_ptr<float>()[0];
  } else {
    throw std::runtime_error("Unknown policy type");
  }

  return value;
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