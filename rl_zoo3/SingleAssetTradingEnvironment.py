from typing import Optional

import gymnasium as gym
import numpy as np
from rl_zoo3.DataGetter import DataGetter

# Environment parameters
STATE_SPACE = 40
ACTION_SPACE = 3

# Action space bounds
ACTION_LOW = -1
ACTION_HIGH = 1

# Trading parameters
COST = 3e-4
CAPITAL = 1000
NEG_MUL = 2


class SingleAssetTradingEnvironment(gym.Env):
    """
    Trading Environment for trading a single asset.
    The Agent interacts with the environment class through the step() function.
    Action Space: {0: Do Nothing, 1: Buy, 2: Sell}
    """

    def __init__(
            self,
            asset_data: DataGetter,
            initial_money=CAPITAL,
            trans_cost=COST,
            store_flag=1,
            asset_ph=0,
            capital_frac=0.2,
            running_thresh=0.1,
            cap_thresh=0.3,
    ):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_SPACE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n=ACTION_SPACE)
        self.past_holding = asset_ph
        self.capital_frac = capital_frac  # Fraction of capital to invest each time.
        self.cap_thresh = cap_thresh
        self.running_thresh = running_thresh
        self.trans_cost = trans_cost

        self.asset_data: DataGetter = asset_data
        self.terminal_idx = len(self.asset_data) - 1
        self.scaler = self.asset_data.scaler

        self.initial_cap = initial_money

        self.capital = self.initial_cap
        self.running_capital = self.capital
        self.asset_inv = self.past_holding

        self.pointer = 0
        self.next_return, self.current_state = 0, None
        self.prev_act = 0
        self.current_act = 0
        self.current_reward = 0
        self.current_price = self.asset_data.frame.iloc[self.pointer, :]["close"]
        self.done = False

        self.store_flag = store_flag
        if self.store_flag == 1:
            self.store = {
                "action_store": [],
                "reward_store": [],
                "running_capital": [],
                "port_ret": [],
            }

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.capital = self.initial_cap
        self.running_capital = self.capital
        self.asset_inv = self.past_holding

        self.pointer = 0
        self.next_return, self.current_state = self.get_state(self.pointer)
        self.prev_act = 0
        self.current_act = 0
        self.current_reward = 0
        self.current_price = self.asset_data.frame.iloc[self.pointer, :]["close"]
        self.done = False

        if self.store_flag == 1:
            self.store = {
                "action_store": [],
                "reward_store": [],
                "running_capital": [],
                "port_ret": [],
            }

        return np.array(self.current_state[0][0], dtype=np.float32), {}

    def step(self, action):
        self.current_act = action
        # print(action)
        self.current_price = self.asset_data.frame.iloc[self.pointer, :]["close"]
        self.current_reward = self.calculate_reward()
        self.prev_act = self.current_act
        self.pointer += 1
        self.next_return, self.current_state = self.get_state(self.pointer)
        self.done = self.check_terminal()

        if self.done:
            reward_offset = 0
            ret = (
                          self.store["running_capital"][-1] / self.store["running_capital"][-0]
                  ) - 1
            if self.pointer < self.terminal_idx:
                reward_offset += -1 * max(0.5, 1 - self.pointer / self.terminal_idx)
            if self.store_flag:
                reward_offset += 10 * ret
            self.current_reward += reward_offset

        if self.store_flag:
            self.store["action_store"].append(self.current_act)
            self.store["reward_store"].append(self.current_reward)
            self.store["running_capital"].append(self.capital)
            info = self.store
        else:
            info = None

        return np.array(self.current_state, dtype=np.float32), self.current_reward, self.done, False, {}

    def calculate_reward(self):
        """
        Initial Setup:

        investment: Calculated as the product of the running capital and a specified fraction (capital_frac), representing the amount of capital to invest in each trade.
        reward_offset: Initialized to zero, but used to adjust the reward in certain cases.
        Buy Action:

        If the current action (current_act) is to buy (1), and if the running capital is sufficient (greater than a threshold), the agent invests in the asset.
        The running capital is reduced by the investment amount.
        The number of asset units purchased is calculated based on the investment and the current price.
        Transaction costs are subtracted from the running capital.
        Sell Action:

        If the current action is to sell (-1), and if the agent holds any units of the asset, the agent sells the asset.
        The running capital is increased by the value of the sold asset, taking into account the current price and transaction costs.
        The number of asset units held by the agent is set to zero.
        Do Nothing:

        If the current action is to do nothing (0), and if the previous action was also to do nothing, a small negative offset (-0.1) is added to the reward (reward_offset).
        Otherwise, no action is taken.
        Final Reward Calculation:

        The total capital (capital) is updated based on the running capital and the value of asset units held.
        The reward is calculated based on various factors:
        The next return (next_return) multiplied by the current action (buying or selling).
        A penalty for frequent trading, represented by the absolute difference between the current and previous actions multiplied by transaction costs.
        If a store flag is set, the portfolio return is stored in a data structure for tracking.
        If the reward is negative, it is multiplied by a constant (NEG_MUL) to make the agent more risk-averse towards negative returns.
        Finally, the reward offset is added to the reward.
        """
        investment = self.running_capital * self.capital_frac
        reward_offset = 0

        # Buy Action
        if self.current_act == 1:
            if self.running_capital > self.initial_cap * self.running_thresh:
                self.running_capital -= investment
                asset_units = investment / self.current_price
                self.asset_inv += asset_units
                self.current_price *= 1 - self.trans_cost

        # Sell Action
        elif self.current_act == -1:
            if self.asset_inv > 0:
                self.running_capital += (
                    self.asset_inv * self.current_price * (1 - self.trans_cost)
                )
                self.asset_inv = 0

        # Do Nothing
        elif self.current_act == 0:
            if self.prev_act == 0:
                reward_offset += -0.1
            pass

        # Reward to give
        prev_cap = self.capital
        self.capital = self.running_capital + (self.asset_inv * self.current_price)
        reward = (
            100 * (self.next_return * self.current_act)
            - np.abs(self.current_act - self.prev_act) * self.trans_cost
        )
        if self.store_flag == 1:
            self.store["port_ret"].append((self.capital - prev_cap) / prev_cap)

        if reward < 0:
            reward *= (
                NEG_MUL  # To make the Agent more risk averse towards negative returns.
            )
        reward += reward_offset

        return reward

    def check_terminal(self):
        if self.pointer == self.terminal_idx:
            return True
        elif self.capital <= self.initial_cap * self.cap_thresh:
            return True
        else:
            return False

    def get_state(self, idx):
        state = self.asset_data[idx][1:]

        state = self.scaler.transform(state.reshape(1, -1))

        state = np.concatenate(
            [
                state,
                [
                    [
                        self.capital / self.initial_cap,
                        self.running_capital / self.capital,
                        self.asset_inv * self.current_price / self.initial_cap,
                        self.prev_act,
                    ]
                ],
            ],
            axis=-1,
        )
        # state_ = pd.DataFrame(state)
        # state_.to_csv("df_saved.csv", index= False)

        next_ret = self.asset_data[idx][0]
        return next_ret, state


print("gym site-packages: " + gym.__file__)
