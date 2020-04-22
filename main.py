import tensorflow as tf
import numpy as np
import gym
import random

from collections import deque

q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])
q_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

target_network = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])
target_network.set_weights(q_network.get_weights())


env = gym.make('CartPole-v1')
memory = deque(maxlen=2000)

epsilon = 0.2


def train():
    batch = np.array(random.sample(memory, 32))

    s = batch[:, 0]
    a = batch[:, 1]
    r = batch[:, 2]
    s2 = batch[:, 3]
    d = batch[:, 4]

    target = q_network.predict(s)
    next_target = target_network.predict(s2)
    selected_next_target = q_network.predict(s2).argmax()
    
    print(target)
    print(next_target)
    print(selected_next_target)



while True:
    state = env.reset().reshape([1, 4])
    done = False
    score = 0

    while True:
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = q_network.predict(state).argmax()
        
        old_state = state
        state, reward, done, _ = env.step(action)
        state = np.reshape(state, [1, 4])
        memory.append([old_state, action, reward, state, done])
        score += reward

        if len(memory) > 32:
            train()

        if done:
            break
    
    target_network.set_weights(q_network.get_weights())
    print(f'Score: {score}')
