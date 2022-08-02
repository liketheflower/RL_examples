import gym

from stable_baselines3 import PPO
# Check model performance
# load the best model you observed from tensorboard - the one reach the goal/ obtaining highest return

models_dir = "models/Mountain-1653282767.3143597"
model_path = f"{models_dir}/80000"
best_model = PPO.load(model_path, env=env)

obs = env.reset()
while True:
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
