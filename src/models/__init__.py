"""
Public API for neural function approximators used in reinforcement-learning
contexts, exposing the canonical feed-forward Q-value network architecture.
"""

from .q_network import QNetwork

__all__ = [
    "QNetwork",
]