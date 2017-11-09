import gym
# import numpy as np
# import matplotlib

# print all the environments
# for e in gym.envs.registry.all():
#     print(e)

env = gym.make('CartPole-v0')
#print(dir(env))

# env.__dict__

observation = env.reset()
for t in range(500):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print ("action:", action, "\t observation:", observation, '\treward:',reward, '\tdone:', done)
    if done:
        break