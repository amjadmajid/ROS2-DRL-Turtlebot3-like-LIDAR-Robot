import numpy as np
import random
import collections
import math
import time

from tensorflow.python.ops.gen_array_ops import expand_dims


class ReplayBuffer:
    buffer = 0

    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def append_sample(self, state, action, reward, next_state, done):
        if self.buffer == 0:
            print("error: appending to uninitialized replay buffer!")
            return
        self.buffer.append([state, action, reward, next_state, done])

    def get_sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_length(self):
        return len(self.buffer)
