import tensorflow as tf
import gym

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

