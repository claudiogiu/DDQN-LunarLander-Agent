from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    """
    Interface for reinforcement-learning agents that operate within environments
    driven by state transitions and reward signals, providing a unified contract
    for action selection, experience processing, and persistence of model state.

    Methods:
        act(state: np.ndarray, eps: float = 0.0) -> int:
            Returns an action given the current environment state, optionally
            modulated by an exploration parameter controlling stochasticity.

        step(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
            Processes a full transition tuple and updates internal learning
            components such as replay buffers, target networks, or policy
            parameters.

        save(path: str) -> None:
            Stores the agent's learnable components and auxiliary state to the
            specified filesystem path.

        load(path: str) -> None:
            Restores the agent's previously saved state from disk, enabling
            evaluation or continuation of training.
    """

    @abstractmethod
    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        pass

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass