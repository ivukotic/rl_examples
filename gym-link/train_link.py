import gym
import gym_link

env = gym.make('Link-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)

for i_episode in range(5):
    observation = env.reset()
    tot_reward = 0
    for t in range(100):
        # env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('obs:', observation, '\trew:', reward, '\tdone:', done)
        tot_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print("total rewards", tot_reward)
            break
