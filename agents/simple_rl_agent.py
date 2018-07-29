from abc import ABCMeta, abstractmethod
import numpy as np
import ipdb

class Agent(metaclass=ABCMeta):
    """Abstract Agent Class"""

    def __init__(self, alpha=0.2, policy=None):
        """
        :param alpha:
        :param policy:
        """
        self.alpha = alpha
        self.policy = policy
        self.reward_history = []

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def get_reward(self, reward):
        pass


class SimpleRLAgent(Agent):
    """
        シンプルな強化学習エージェント
    """
    def __init__(self, action_list=None, **kwargs):
        """
        :param action_list:
        :param q_values:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.action_list = action_list  # 選択肢
        self.last_action_id = None
        self.q_values = self._init_q_values()   # 期待報酬値の初期化

    def _init_q_values(self):
        q_values = {}
        q_values = np.repeat(0.0, len(self.action_list))
        return q_values

    def act(self, q_values=None):
        if self.policy.name == "UCB":
            action_id = self.policy.select_action()    # 行動選択
        else:
            action_id = self.policy.select_action(self.q_values)    # 行動選択
        self.last_action_id = action_id
        action = self.action_list[action_id]
        return action

    def get_reward(self, reward):
        self.reward_history.append(reward)
        self.q_values[self.last_action_id] = self._update_q_value(reward)   # 期待報酬値の更新
        if self.policy.name == "UCB":
            self.policy.update_average_rewards(self.last_action_id, reward) 
            self.policy.update_usbs(self.last_action_id, reward)


    def _update_q_value(self, reward):
        return ((1.0 - self.alpha) * self.q_values[self.last_action_id]) + (self.alpha * reward) # 通常の指数移動平均で更新



INIQ = 0.0    # 初期のQ値
MinAlpha = 0.01
MinEps = 0.01
class QLearningAgent(Agent):
    """
        シンプルなq学習エージェント
    """
    def __init__(self, gamma=0.99, actions=None, observation=None, alpha_decay_rate=None, epsilon_decay_rate=None,**kwargs):
        super().__init__(**kwargs)
        self.name = "qlearning"
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.state = str(observation)
        self.ini_state = str(observation)
        self.last_state = str(observation)
        self.last_action_id = None
        self.q_values = self._init_q_values()
        self.is_share = False

    def _init_q_values(self):
        q_values = {}
        q_values[self.state] = np.repeat(INIQ, len(self.actions))
        return q_values

    def init_state(self):
        self.last_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def init_policy(self, policy):
        self.policy = policy

    def act(self, q_values=None, step=0):
        action_id = self.policy.select_action(self.q_values[self.state])
        self.last_action_id = action_id
        action = self.actions[action_id]
        return action

    def get_reward(self, reward, is_finish=True, step=0):
        self.reward_history.append(reward)
        self.q_values[self.last_state][self.last_action_id], pre_q, max_q2 = self._update_q_value(reward)

    def observe(self, next_state):
        next_state = str(next_state)
        if next_state not in self.q_values:
            self.q_values[next_state] = np.repeat(INIQ, len(self.actions))

        self.last_state = self.state
        self.state = next_state

    def _update_q_value(self, reward):
        try:
            pre_q = self.q_values[self.last_state][self.last_action_id]
            max_q2 = max(self.q_values[self.state])
            updated_q = ((1.0 - self.alpha) * pre_q) + (self.alpha * (reward + (self.gamma * max_q2)))
        except:
            ipdb.set_trace()

        return updated_q, pre_q, max_q2

    def get_data(self):
        result = {}
        result["alpha"] = self.alpha
        result["gamma"] = self.gamma
        result["epsilon"] = self.policy.eps
        result["epsilon_log"] = self.policy.get_log()
        result["reward_history"] = self.reward_history
        return result

    def decay_alpha(self):
        if self.alpha_decay_rate is not None:
            if self.alpha >= MinAlpha:
                self.alpha *= self.alpha_decay_rate

    def decay_epsilon(self):
        if self.epsilon_decay_rate is not None:
            if self.policy.epsilon >= MinEps:
                self.policy.epsilon *= self.epsilon_decay_rate
