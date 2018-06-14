import random
import numpy as np
import matplotlib.pyplot as plt

class Arm():
    def __init__(self, conf):
        self.mu = conf["mu"]
        self.sd = conf["sd"]

    def pull(self):
        return np.random.normal(self.mu, self.sd)

class MultiArmBandit():
    def __init__(self, conf_arm):
        self.arms = self._init_arms(conf_arm)

    def _init_arms(self, conf_arm):
        arms = []
        for cf in conf_arm:
            arms.append(Arm(cf))

        return arms

    def step(self, arm_id):
        """
            pull lever 
        """
        return self.arms[arm_id].pull()

if __name__ == '__main__':
    conf_arm = [{"id":0, "mu":0.1, "sd":1}, 
                {"id":1, "mu":0.5, "sd":1}, 
                {"id":2, "mu":1, "sd":1}, 
                {"id":3, "mu":0.8, "sd":1}, 
                {"id":4, "mu":0.9, "sd":1}, 
                ]

    game = MultiArmBandit(conf_arm=conf_arm) # 5本のアームを設定
    nb_step = 100   #ステップ数
    rewards = []
    for step in range(nb_step):
        reward = game.step(random.randint(0, 4))
        rewards.append(reward)

    plt.plot(np.arange(nb_step), rewards)
    # plt.ylim(0, 1)
    plt.show()
    plt.savefig("result1.png")
