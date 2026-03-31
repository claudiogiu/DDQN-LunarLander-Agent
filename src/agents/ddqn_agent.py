import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from agents.base_agent import BaseAgent
from models.q_network import QNetwork
from memory.replay_buffer import ReplayBuffer

class DDQNAgent(BaseAgent):
    """
    Interface for Double Deep Q-Network agents implementing value-based control
    with decoupled action selection and evaluation, experience replay, and
    soft-updated target networks for stabilizing temporal-difference learning.

    Attributes:
        state_size (int): Dimensionality of the environment state vector.
        action_size (int): Number of discrete actions available to the agent.
        gamma (float): Discount factor applied to future rewards.
        tau (float): Interpolation coefficient for soft updates of target parameters.
        batch_size (int): Number of transitions sampled per learning update.
        update_every (int): Frequency (in environment steps) of learning updates.
        epsilon (float): Current ε value for ε-greedy exploration.
        epsilon_end (float): Minimum exploration probability.
        epsilon_decay (float): Multiplicative decay factor applied to ε after each update.
        device (torch.device): Compute device used for neural network inference and optimization.
        qnetwork_local (QNetwork): Online Q-function approximator used for action selection.
        qnetwork_target (QNetwork): Target Q-function approximator providing stable TD targets.
        optimizer (torch.optim.Optimizer): Optimizer updating the local network parameters.
        memory (ReplayBuffer): Fixed-capacity buffer storing transitions for uniform sampling.
        _t_step (int): Internal counter tracking environment steps for update scheduling.

    Methods:
        act(state: np.ndarray, eps: float = None) -> int:
            Returns an action selected via ε-greedy policy using the local Q-network.

        step(state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
            Stores a transition in memory and triggers learning updates at
            the configured frequency.

        _learn(experiences) -> None:
            Performs a DDQN update by computing TD targets using the target
            network and updating the local network via MSE loss.

        _soft_update(local_model, target_model) -> None:
            Applies a soft parameter update to the target network using τ.

        _update_epsilon() -> None:
            Decays ε according to the configured schedule.

        save(path: str) -> None:
            Serializes agent parameters, network weights, and optimizer state.

        load(path: str) -> None:
            Restores agent parameters, network weights, and optimizer state.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 128,
        gamma: float = 0.99,
        tau: float = 1e-2,
        update_every: int = 4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.999
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(buffer_size)
        self._t_step = 0


    def act(self, state: np.ndarray, eps: float = None) -> int:
        if eps is None:
            eps = self.epsilon

        if random.random() < eps:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
        return int(torch.argmax(q_values).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) < 2000:
            return

        self._t_step += 1
        if self._t_step % self.update_every == 0:
            experiences = self.memory.sample(self.batch_size)
            self._learn(experiences)
            self._update_epsilon()

    def _learn(self, experiences) -> None:
        states, actions, rewards, next_states, dones = experiences

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_q_local = self.qnetwork_local(next_states)
            next_actions = torch.argmax(next_q_local, dim=1, keepdim=True)

            next_q_target = self.qnetwork_target(next_states)
            q_targets_next = next_q_target.gather(1, next_actions)

            q_targets = rewards + self.gamma * q_targets_next * (1 - dones)

        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target)

    def _soft_update(self, local_model, target_model) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def _update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        checkpoint = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "qnetwork_local": self.qnetwork_local.state_dict(),
            "qnetwork_target": self.qnetwork_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.qnetwork_local.load_state_dict(checkpoint["qnetwork_local"])
        self.qnetwork_target.load_state_dict(checkpoint["qnetwork_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]