#pragma once

#include <string>
#include <torch/script.h>

namespace baselines3_models {

torch::Tensor multi_one_hot(torch::Tensor &input, torch::Tensor &classes);

}