import random
import numpy as np
import matplotlib.pyplot as plt

class Arm():
    def __init__(self, idx):
        self.value = random.random()    # ランダムでこのアームを引いた時の報酬を設定

    def pull(self):
        return self.value

class MultiArmBandit():
    def __init__(self, nb_arm):
        self.arms = self._init_arms(nb_arm)

    def _init_arms(self, nb_arm):
        arms = []
        for i in range(nb_arm):
            arms.append(Arm(i))

        return arms

    def step(self, arm_id):
        """
            pull lever 
        """
        return self.arms[arm_id].pull()

if __name__ == '__main__':
    game = MultiArmBandit(nb_arm=5) # 5本のアームを設定
    nb_step = 100   #ステップ数
    rewards = []
    for step in range(nb_step):
        reward = game.step(random.randint(0, 4))
        rewards.append(reward)

    plt.plot(np.arange(nb_step), rewards)
    plt.ylim(0, 1)
    plt.savefig("result1.png")
    # plt.show()
