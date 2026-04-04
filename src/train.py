import os
import gymnasium as gym
import warnings
from agents.ddqn_agent import DDQNAgent

warnings.filterwarnings("ignore")

class Trainer:
    """
    Interface for orchestrating episodic training of a DDQN agent on a
    Gymnasium-compatible environment, providing controlled interaction loops,
    episodic return monitoring, and final policy serialization.

    Attributes:
        n_episodes (int): Total number of training episodes executed sequentially.
        max_t (int): Maximum number of environment steps allowed per episode.
        env (gym.Env): Environment instance used for agent-environment interaction.
        state_size (int): Dimensionality of the observation vector returned by the environment.
        action_size (int): Cardinality of the discrete action space.
        agent (DDQNAgent): Deep Double Q-Network agent responsible for action selection and value-function updates.

    Methods:
        train() -> None:
            Executes the episodic training loop, collecting transitions,
            updating the agent, and reporting per-episode returns.
    """

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        n_episodes: int = 1000,
        max_t: int = 1000
    ) -> None:

        self.env = gym.make(env_name)
        self.n_episodes = n_episodes
        self.max_t = max_t

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.agent = DDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=0
        )

        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    @property
    def results_dir(self) -> str:
        return os.path.join(self.project_root, "results")

    @property
    def model_path(self) -> str:
        return os.path.join(self.results_dir, "ddqn_lunarlander.pt")

    def train(self) -> None:
        print(f"Starting DDQN policy learning for {self.n_episodes} episodes...\n")
        for episode in range(1, self.n_episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0.0

            for _ in range(self.max_t):
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.agent.step(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

            print(f"Episode {episode} - Return: {total_reward:.2f}")
        
        self.agent.save(self.model_path)
        print(f"Model saved to: {self.model_path}")
        self.env.close()


if __name__ == "__main__":
    trainer: Trainer = Trainer()
    trainer.train()
