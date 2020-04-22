import tensorflow as tf
import numpy as np
import gym
import random
import pickle
import os

from collections import deque


# Root path
PATH = './'

# Model path
MODEL_PATH = PATH + 'model/'
MEMORY_PATH = PATH + 'memory/'
MODEL_FILE = MODEL_PATH + 'target_network.h5'
MEMORY_FILE = MEMORY_PATH + 'dequed_memory.pickle'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

if not os.path.exists(MEMORY_PATH):
    os.mkdir(MEMORY_PATH)

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


def train():
    batch = random.sample(memory, 32)
    
    s, a, r, s2, d = [], [], [], [], []
    for i in batch:
        s.append(i[0])
        a.append(i[1])
        r.append(i[2])
        s2.append(i[3])
        d.append(i[4])
    
    s = np.array(s)
    a = np.array(a)
    r = np.array(r)
    s2 = np.array(s2)
    d = np.array(d)

    target = q_network.predict(s)
    next_target = target_network.predict(s2)
    selected_next_target = q_network.predict(s2).argmax(axis=1)

    for i in range(32):
        if not d[i]:
            target[i] = r[i] + gamma * next_target[i][selected_next_target[i]]
        else:
            target[i] = r[i]

    q_network.fit(s, target, epochs=1, verbose=0)



while True:
    state = env.reset()
    done = False
    score = 0

    while True:
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = q_network.predict(np.reshape(state, [1, 4])).argmax()
        
        old_state = state
        state, reward, done, _ = env.step(action)
        memory.append([old_state, action, reward, state, done])
        score += reward

        if len(memory) > 32:
            train()

        if done:
            break
    
    target_network.set_weights(q_network.get_weights())
    target_network.save(MODEL_FILE)
    with open(MEMORY_FILE, 'wb') as file:
        pickle.dump(memory, file)
    print(f'Score: {score}')
