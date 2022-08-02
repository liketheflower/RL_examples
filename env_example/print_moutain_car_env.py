import gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
print("MoutainCar env")
print(env)
print("action_space")
print(env.action_space)
print("observation_space")
print(env.observation_space)
print("reward_range")
print(env.reward_range)
