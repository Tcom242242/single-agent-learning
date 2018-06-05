import os,sys
sys.path.append(os.getcwd())
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.sinple_rl_agent import SimpleRLAgent
from agents.policy import EpsGreedyQPolicy
from envs.multi_arm_bandit import MultiArmBandit

if __name__ == '__main__':
    nb_arm=5    # armの数
    game = MultiArmBandit(nb_arm=nb_arm) # 5本のアームを設定
    policy = EpsGreedyQPolicy(epsilon=0.001)    # exploration rate を0.001に設定
    agent = SimpleRLAgent(alpha=0.1, policy=policy, action_list=np.arange(nb_arm))  # agentの設定
    nb_step = 10000   #ステップ数
    rewards = []
    for step in range(nb_step):
        action = agent.act()    # レバーの選択
        reward = game.step(action) # レバーを引く
        rewards.append(reward) 
        agent.get_reward(reward) #　エージェントは報酬を受け取り学習

    plt.plot(np.arange(nb_step), rewards)
    plt.ylim(0, 1)
    plt.ylabel("reward")
    plt.xlabel("steps")
    plt.savefig("fig/result2.png")
    plt.show()
