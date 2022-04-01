// This file is just a demonstration, you can adapt to test your model
// First, include your model:
#include "baselines3_models/cartpole_v1.h"

using namespace baselines3_models;

int main(int argc, const char *argv[]) {  
  // Create an instance of it:
  CartPole_v1 cartpole;

  // Build an observation:
  torch::Tensor observation = torch::tensor({0., 0., 0., 0.});

  // You can now check the prediction:
  std::cout << cartpole.predict(observation) << std::endl;
}