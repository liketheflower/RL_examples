from stable_baselines3 import PPO
import gym

env = gym.make("CartPole-v1")

model = PPO(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=25000)

model.save("ppo_cartpole")  # saving the model to ppo_cartpole.zip
model = PPO.load("ppo_cartpole")  # loading the model from ppo_cartpole.zip

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
