import gym
import gym_link

import math
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class LinkSolver():
    def __init__(self, n_episodes=1000, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, lr=0.01, lr_decay=0.01, batch_size=64):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('Link-v0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=3, activation='relu'))
        # self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(4, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr, decay=self.lr_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 3])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        avg_reward = deque(maxlen=100)
        avg_life = deque(maxlen=100)
        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            tot_reward = 0
            steps = 0
            info = None
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, info = self.env.step(action)
                # self.env.render()
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                tot_reward += reward
                steps += 1
                # if steps % 200 == 0:
                #    print('episode:', e, '\tsteps:', steps)

            avg_reward.append(tot_reward)
            avg_life.append(steps)

            mean_reward = np.mean(avg_reward)
            mean_life = np.mean(avg_life)

            print('duration:{}    reward:{}'.format(steps, tot_reward), info)

            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} s. Mean reward was {}.'.format(e, mean_life, mean_reward))

            self.replay(self.batch_size)
        return


if __name__ == '__main__':
    agent = LinkSolver()
    agent.run()
