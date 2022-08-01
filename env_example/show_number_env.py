import gym

envs = gym.envs.registry.all()
print(f"In total we have {len(envs)} envs available!")
print(f"The first 4 envs are: ")
for i, env in enumerate(list(envs)[:4]):
    print("-" * 20)
    print(env)
