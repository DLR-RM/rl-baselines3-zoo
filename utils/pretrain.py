from typing import Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.utils import get_device, set_random_seed
from stable_baselines3.common.type_aliases import GymEnv


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations: np.ndarray, expert_actions: np.ndarray):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.observations[index], self.actions[index]

    def __len__(self) -> int:
        return len(self.observations)


def prepare_data(expert_data_path: str,
                 train_size: float = 0.8) -> Tuple[ExpertDataSet, ExpertDataSet]:
    data = np.load(expert_data_path)
    expert_dataset = ExpertDataSet(data['expert_observations'], data['expert_actions'])

    train_size = int(train_size * len(expert_dataset))

    test_size = len(expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    print(f"test_expert_dataset: {len(test_expert_dataset)} entries")
    print(f"train_expert_dataset: {len(train_expert_dataset)} entries")
    return train_expert_dataset, test_expert_dataset


def predict(student: Union[A2C, PPO, SAC, TD3],
            model: nn.Module,
            env: GymEnv,
            data: th.Tensor,
            target: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    if isinstance(env.action_space, gym.spaces.Box):
        # A2C/PPO policy outputs actions, values, log_prob
        # SAC/TD3 policy outputs actions only
        if isinstance(student, (A2C, PPO)):
            action, _, _ = model(data)
        else:
            # SAC/TD3:
            action = model(data)
        action_prediction = action.double()
    else:
        # Retrieve the logits for A2C/PPO when using discrete actions
        latent_pi, _, _ = model._get_latent(data)
        logits = model.action_net(latent_pi)
        action_prediction = logits
        target = target.long()
    return action_prediction, target


def pretrain_agent(expert_data_path: str,
                   student: Union[A2C, PPO, SAC, TD3],
                   env: GymEnv,
                   train_size: float = 0.8,
                   batch_size: int = 64,
                   n_epochs: int = 100,
                   lr_decay: float = 1.0,
                   learning_rate: float = 3e-4,
                   log_interval: int = 1,
                   seed: int = 1,
                   test_batch_size: int = 64) -> Union[A2C, PPO, SAC, TD3]:
    set_random_seed(seed)

    train_expert_dataset, test_expert_dataset = prepare_data(expert_data_path, train_size)

    device = get_device('auto')
    kwargs = {"num_workers": 1, "pin_memory": True} if device == th.device('cuda') else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = DataLoader(dataset=train_expert_dataset, batch_size=batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_expert_dataset, batch_size=test_batch_size,
                             shuffle=True, **kwargs)

    # Define an Optimizer and a learning rate schedule.
    optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            action_prediction, target = predict(student, model, env, data, target)

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
        if epoch % log_interval == 0:
            print(f"Train Epoch: {epoch}/{n_epochs} \tLoss: {loss.item():.6f}")

        # Test
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                action_prediction, target = predict(student, model, env, data, target)

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.6f}")

        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model
    return student
