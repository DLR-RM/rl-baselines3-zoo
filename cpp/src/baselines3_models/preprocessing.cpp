#include "baselines3_models/preprocessing.h"

using namespace torch::indexing;

namespace baselines3_models {

torch::Tensor multi_one_hot(torch::Tensor &input, torch::Tensor &classes) {
  int entries = torch::sum(classes).item<int>();

  torch::Tensor result =
      torch::zeros({1, entries}, torch::TensorOptions().dtype(torch::kLong));

  int offset = 0;
  for (int k = 0; k < classes.sizes()[0]; k++) {
    int n = classes[k].item<int>();

    result.index({0, Slice(offset, offset + n)}) = torch::one_hot(input[k], n);
    offset += n;
  }

  return result;
}

} // namespace baselines3_models