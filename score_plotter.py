import matplotlib.pyplot as plt

# Root path
PATH = './'

# Folder path
MEMORY_PATH = PATH + 'memory/'

# File path
HISTORY_FILE = MEMORY_PATH + 'history.txt'

with open(HISTORY_FILE, 'r') as f:
    content = [float(x.strip()) for x in f.readlines()]


plt.figure(num='DDQN TAXI Score History')


y = [i for i in content]
plt.plot(y)

plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.show()
