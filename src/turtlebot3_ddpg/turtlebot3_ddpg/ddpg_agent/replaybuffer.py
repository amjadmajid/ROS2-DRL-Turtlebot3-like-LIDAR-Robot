import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size

    def sample(self, count):
        batch = []
        count = min(count, self.get_length())
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def get_length(self):
        return len(self.buffer)

    def add_sample(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.buffer.append(transition)
