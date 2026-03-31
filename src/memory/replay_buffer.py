import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Interface for fixed-capacity experience storage supporting uniform sampling
    of transition tuples, providing a cyclic memory structure for stabilizing
    value-function learning through decorrelated mini-batches.

    Methods:
        push(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
            Inserts a transition into the buffer, overwriting the oldest entry
            once the maximum capacity is reached.

        sample(batch_size: int):
            Draws a uniformly random mini-batch of transitions and returns
            tensors for states, actions, rewards, next states, and terminal flags.

        __len__() -> int:
            Returns the current number of stored transitions.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buffer = []
        self._position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        transition = (state, action, reward, next_state, done)

        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition

        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, batch_size)

        states = torch.tensor(
            np.vstack([b[0] for b in batch]), dtype=torch.float32
        )
        actions = torch.tensor(
            np.array([b[1] for b in batch]), dtype=torch.int64
        ).unsqueeze(1)
        rewards = torch.tensor(
            np.array([b[2] for b in batch]), dtype=torch.float32
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.vstack([b[3] for b in batch]), dtype=torch.float32
        )
        dones = torch.tensor(
            np.array([b[4] for b in batch]), dtype=torch.float32
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)