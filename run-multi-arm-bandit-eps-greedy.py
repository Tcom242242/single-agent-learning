import os,sys
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.simple_rl_agent import SimpleRLAgent
from agents.policy import EpsGreedyQPolicy, UCB
from envs.multi_arm_bandit import MultiArmBandit

if __name__ == '__main__':
    conf_arm = [{"id":0, "mu":0.1, "sd":1}, 
                {"id":1, "mu":0.5, "sd":1}, 
                {"id":2, "mu":1, "sd":0.1}, 
                {"id":3, "mu":0.2, "sd":1}, 
                {"id":4, "mu":0.4, "sd":1}, 
                ]

    game = MultiArmBandit(conf_arm=conf_arm) # 5本のアームを設定
    # policy = EpsGreedyQPolicy(epsilon=0.001)    # exploration rate を0.001に設定
    policy = UCB(c=0.2, actions=np.arange(len(conf_arm)))    # exploration rate を0.001に設定
    agent = SimpleRLAgent(alpha=0.1, policy=policy, action_list=np.arange(len(conf_arm)))  # agentの設定
    nb_step = 10000   #ステップ数
    rewards = []
    for step in range(nb_step):
        action = agent.act()    # レバーの選択
        reward = game.step(action) # レバーを引く
        rewards.append(reward) 
        agent.get_reward(reward) #　エージェントは報酬を受け取り学習
        # print(agent.policy.ucbs)

    plt.plot(np.arange(nb_step), rewards)
    # plt.ylim(0, 1)
    plt.ylabel("reward")
    plt.xlabel("steps")
    plt.savefig("fig/result2.png")
    plt.show()
