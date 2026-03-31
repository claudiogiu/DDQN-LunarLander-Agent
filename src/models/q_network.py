import torch
from torch import nn

class QNetwork(nn.Module):
    """
    Interface for feed-forward neural architectures used to approximate action-value
    functions in reinforcement-learning settings, providing a compact multilayer
    mapping from continuous state representations to discrete action scores.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the network's output scores for the given batch of input
            states, returning one Q-value per available action.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)