import random
import numpy as np
import matplotlib.pyplot as plt
import ipdb

FILED_TIPE = {
    "N":0, 
    "S":1, 
    "G":2, 
    }

ACTIONS = {
    "STAY": 0, 
    "UP": 1, 
    "DOWN": 2, 
    "LEFT": 3, 
    "RIGHT":4
    }

class GridEnv():
    def __init__(self):

        self.map = [[0, 0, 0, 2], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [1, 0, 0, 0]]

        self.start_pos = 4, 0
        self.agent_pos = 4, 0

    def step(self, action):
        """
            return pos, reward
        """
        to_y, to_x = self.agent_pos
        # 移動可能かどうかの確認。移動不可能であれば、ポジションはそのままにマイナス報酬@todo
        if self._is_possible_action(to_x, to_y, action) == False:
            return self.agent_pos, -1, False

        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1

        is_goal = self._is_goal(to_x, to_y)
        reward = self._compute_reward(to_x, to_y) # @todo
        self.agent_pos = to_y, to_x
        return self.agent_pos, reward, is_goal

    def _is_goal(self, x, y):
        if self.map[y][x] == FILED_TIPE["G"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """ 
            実行可能な行動かどうかの判定
        """
        to_x = x
        to_y = y

        if action == ACTIONS["STAY"]:
            return True
        elif action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1
        else:
            raize("Action Eroor")

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == FILED_TIPE["N"] or self.map[y][x] == FILED_TIPE["S"]:
            return 0
        elif self.map[y][x] == FILED_TIPE["G"]:
            return 1

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos


if __name__ == '__main__':
    grid_env = GridEnv() # 5本のアームを設定
    ini_state = grid_env.start_pos
    is_goal = False
    agent = QLearningAgent(actions=np.arange(4), observation=ini_state)
    nb_episode = 100   #ステップ数
    for episode in range(nb_episode): 
        while(is_goal == False):
            action = agent.act()
            state, reward, is_goal = grid_env.step(action)
            agent.observe(state)
            agent.get_reward(reward)

        rewar = game.step(random.randint(0, 4))
        rewards.append(reward)

    plt.plot(np.arange(nb_step), rewards)
    # plt.ylim(0, 1)
    plt.show()
    plt.savefig("result1.png")
    pass
