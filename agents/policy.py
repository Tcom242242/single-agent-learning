import copy
import numpy as np
import math
import ipdb
from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):

    @abstractmethod
    def select_action(self, **kwargs):
        pass

class EpsGreedyQPolicy(Policy):
    """
        ε-greedy選択 
    """
    def __init__(self, epsilon=.1, decay_rate=1):
        super(EpsGreedyQPolicy, self).__init__()
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.name = "EPS"

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.random_integers(0, nb_actions-1)
        else:   # greedy 行動
            action = np.argmax(q_values)

        return action

    def decay_epsilon():    # 探索率を減少
        self.epsilon = self.epsilon*self.decay_rate


class UCB(Policy):
    def __init__(self, c, actions):
        super(UCB, self).__init__()
        self.average_rewards = np.repeat(0.0, len(actions)) # 各腕の平均報酬
        self.ucbs = np.repeat(10.0, len(actions))   # UCB値
        self.counters = np.repeat(0, len(actions))  # 各腕の試行回数
        self.c = c
        self.all_conter = 0 # 全試行回数
        self.name = "UCB"

    def select_action(self):
        action_id = np.argmax(self.ucbs)
        self.counters[action_id] += 1
        self.all_conter += 1
        return action_id

    def update_usbs(self, action_id, reward):
        self.ucbs[action_id] = self.average_rewards[action_id] + self.c * math.log(self.all_conter)/np.sqrt(self.counters[action_id])

    def update_average_rewards(self, action_id, reward):
        self.average_rewards[action_id] = (self.counters[action_id]*self.average_rewards[action_id]+reward)/(self.counters[action_id]+1)

