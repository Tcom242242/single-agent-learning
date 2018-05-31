import os,sys
sys.path.append(os.getcwd())
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.sinple_rl_agent import SimpleRLAgent
from agents.policy import EpsGreedyQPolicy

from envs.multi_arm_bandit import MultiArmBandit

if __name__ == '__main__':
    nb_arm=5
    game = MultiArmBandit(nb_arm=nb_arm) # 5本のアームを設定
    # agent の生成
    policy = EpsGreedyQPolicy(epsilon=0.001)
    agent = SimpleRLAgent(alpha=0.1, policy=policy, action_list=np.arange(nb_arm))
    nb_step = 10000   #ステップ数
    rewards = []
    for step in range(nb_step):
        action = agent.act()
        reward = game.step(action)
        rewards.append(reward)
        agent.get_reward(reward)

    plt.plot(np.arange(nb_step), rewards)
    plt.ylim(0, 1)
    plt.savefig("fig/result2.png")
    plt.show()
