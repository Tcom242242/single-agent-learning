import os,sys
sys.path.append(os.getcwd())
import random
import numpy as np
import matplotlib.pyplot as plt

from envs.multi_arm_bandit import MultiArmBandit

if __name__ == '__main__':
    game = MultiArmBandit(nb_arm=5) # 5本のアームを設定
    nb_step = 100   #ステップ数
    rewards = []
    for step in range(nb_step):
        reward = game.step(random.randint(0, 4))
        rewards.append(reward)

    plt.plot(np.arange(nb_step), rewards)
    plt.ylim(0, 1)
    plt.savefig("fig/result1.png")
    plt.show()
