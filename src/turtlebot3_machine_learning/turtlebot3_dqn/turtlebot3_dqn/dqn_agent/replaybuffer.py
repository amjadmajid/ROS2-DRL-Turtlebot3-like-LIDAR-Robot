import numpy as np
import random
import collections
import math
import time


class ReplayBuffer:
    memory = 0

    def __init__(self, memory_size):
        self.memory = collections.deque(maxlen=memory_size)

    def append_sample(self, state, action, reward, next_state, done):
        if self.memory == 0:
            print("error: appending to uninitialized replay buffer!")
            return
        self.memory.append((state, action, reward, next_state, done))
