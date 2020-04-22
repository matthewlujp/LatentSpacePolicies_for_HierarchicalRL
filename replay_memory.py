import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0
        self.step = 0
        self.full = False

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.full |= self.step >= self.capacity
        self.step += 1

    def reset(self):
        self.buffer = []
        self.position = 0
        self.step = 0
        self.full = False

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward[:, None], next_state, done[:, None]

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
