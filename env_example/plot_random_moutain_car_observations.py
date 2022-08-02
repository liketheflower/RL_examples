import gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()
observations = []
for t in range(1000):
    # env.render()
    observation = env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    observations.append(observation)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
observations = np.array(observations)
plt.scatter(-observations[:, 0], observations[:, 1], alpha=0.3)
plt.savefig("observations.png")
plt.show()
env.close()
