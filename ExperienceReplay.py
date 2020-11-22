from collections import namedtuple
import random
from typing import List
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReservoirReplayMemory(ReplayMemory):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.N = 0

    def push(self, *args):
        if self.capacity > self.N:
            assert self.N == len(
                self.memory), "N should be number of samples inserted so far and when smaller than capacity should be equal to the length of memory."
            self.memory.append(Transition(*args))
        else:
            self.position = random.randint(0, self.N)
            if self.position < self.capacity:
                self.memory[self.position] = Transition(*args)
        self.N += 1
