import gym
from stable_baselines3 import PPO

# Parallel environments
# env = make_vec_env("LunarLander-v2", n_envs=8)

# Create environment
env = gym.make("LunarLander-v2")

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(10))
# Save the agent
#model.save("ppo_lunar2")

# Load the trained agent
# model = PPO.load("ppo_lunar", env=env)

# Enjoy trained agent
obs = env.reset()
for i in range(2):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
