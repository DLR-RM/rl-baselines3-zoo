#include "baselines3_models/FILE_NAME.h"
#include "baselines3_models/preprocessing.h"

namespace baselines3_models {

CLASS_NAME::CLASS_NAME() : Predictor("MODEL_FNAME") {
  policy_type = POLICY_TYPE;
}

torch::Tensor CLASS_NAME::preprocess_observation(torch::Tensor &observation) {
  torch::Tensor result;
  PREPROCESS_OBSERVATION
  return result;
}

torch::Tensor CLASS_NAME::process_action(torch::Tensor &action) {
  torch::Tensor result;
  PROCESS_ACTION
  return result;
}

} // namespace baselines3_models