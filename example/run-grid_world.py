import os,sys
sys.path.append(os.getcwd())
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.simple_rl_agent import QLearningAgent
from agents.policy import EpsGreedyQPolicy, UCB
from envs.grid_env import GridEnv
import ipdb

if __name__ == '__main__':

    grid_env = GridEnv() # 5本のアームを設定
    ini_state = grid_env.start_pos
    is_goal = False
    policy = EpsGreedyQPolicy(epsilon=1.0)
    agent = QLearningAgent(actions=np.arange(5), observation=ini_state, policy=policy, epsilon_decay_rate=0.999)
    nb_episode = 10000   #ステップ数
    rewards = []
    for episode in range(nb_episode): 
        episode_reward = []
        while(is_goal == False):
            action = agent.act()
            state, reward, is_goal = grid_env.step(action)
            # print("action:{}, state:{}, reward:{}".format(action, state, reward))
            agent.observe(state)
            agent.get_reward(reward)
            agent.decay_alpha()
            agent.decay_epsilon()
            episode_reward.append(reward)
            print(agent.policy.epsilon)
        rewards.append(np.mean(episode_reward))
        state = grid_env.reset()
        is_goal = False
        agent.observe(state)

    plt.plot(np.arange(nb_episode), rewards)
    # plt.ylim(0, 1)
    plt.show()
    plt.savefig("result1.png")

