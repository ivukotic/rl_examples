""" test """
import gym
# import numpy as np
# import matplotlib

# print all the environments
# for e in gym.envs.registry.all():
    #  print(e)

dp = gym.make('Pendulum-v0')
print(dp.action_space)
print(dp.observation_space)
print(dp.observation_space.low)
print(dp.observation_space.high)

env = gym.make('CartPole-v1')
print(dp.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)


for i_episode in range(5):
    observation = env.reset()
    tot_reward=0
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        tot_reward+=reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("total rewards", tot_reward)
            break
