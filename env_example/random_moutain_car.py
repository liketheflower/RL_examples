import gym
import time

env = gym.make("MountainCar-v0")
env.reset()
for t in range(10):
    print("-" * 20 + " " + str(t) + " " + "-" * 20)
    env.render()
    observation = env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"observation: {observation}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"info {info}")
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
    time.sleep(1)
env.close()
