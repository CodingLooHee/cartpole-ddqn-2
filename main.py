import tensorflow as tf
import numpy as np
import gym
from random import randint, random, sample
import pickle
import os

from collections import deque


# Root path
PATH = './'

# Folder path
MODEL_PATH = PATH + 'model/'
MEMORY_PATH = PATH + 'memory/'

# File path
MODEL_FILE = MODEL_PATH + 'target_network.h5'
MEMORY_FILE = MEMORY_PATH + 'dequed_memory.pickle'
HISTORY_FILE = MEMORY_PATH + 'history.txt'

if not os.path.exists(PATH):
    os.makedirs(PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(MEMORY_PATH):
    os.makedirs(MEMORY_PATH)

if os.path.exists(MODEL_FILE):
    q_network = tf.keras.models.load_model(MODEL_FILE)
    target_network = tf.keras.models.load_model(MODEL_FILE)
    q_network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
else:
    q_network = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    q_network.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')

    target_network = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    target_network.set_weights(q_network.get_weights())


env = gym.make('CartPole-v1')
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, 'rb') as file:
        memory = pickle.load(file)
else:
    memory = deque(maxlen=2000)


epsilon = 0.2
gamma = 0.95


np_array = np.array
np_reshape = np.reshape

q_network_predict = q_network.predict
q_network_fit = q_network.fit

target_network_predict = target_network.predict

env_step = env.step
env_reset = env.reset

while True:
    state = env_reset()
    done = False
    score = 0

    while True:
        if random() < epsilon:
            action = randint(0, 1)
        else:
            action = q_network_predict(np_reshape(state, [1, 4])).argmax()
        
        old_state = state
        state, reward, done, _ = env_step(action)
        memory.append([old_state, action, reward, state, done])
        score += reward

        if len(memory) > 32:
            batch = sample(memory, 32)
    
            s, a, r, s2, d = [], [], [], [], []
            s_append = s.append
            a_append = a.append
            r_append = r.append
            s2_append = s2.append
            d_append = d.append
            for i in batch:
                s_append(i[0])
                a_append(i[1])
                r_append(i[2])
                s2_append(i[3])
                d_append(i[4])
            
            s = np_array(s)
            a = np_array(a)
            r = np_array(r)
            s2 = np_array(s2)
            d = np_array(d)

            target = q_network_predict(s)
            next_target = target_network_predict(s2)
            selected_next_target = q_network_predict(s2).argmax(axis=1)

            for i in range(32):
                if not d[i]:
                    target[i] = r[i] + gamma * next_target[i][selected_next_target[i]]
                else:
                    target[i] = r[i]

            q_network_fit(s, target, epochs=1, verbose=0)


        if done:
            break
    
    target_network.set_weights(q_network.get_weights())
    target_network.save(MODEL_FILE)
    with open(MEMORY_FILE, 'wb') as file:
        pickle.dump(memory, file)
    with open(HISTORY_FILE, 'a') as file:
        file.write(str(score) + '\n')
    print(f'Score: {score}')
