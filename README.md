# DDQN for the Gymnasium LunarLander Environment

## Introduction  

This repository is designed for training a Double Deep Q‑Network (DDQN) agent on the Gymnasium LunarLander environment. The implemented methodology corresponds to the Double Q‑learning formulation extended to deep neural function approximators, originally introduced by VAN HASSELT H., GUEZ A., and SILVER D. (2016) in their paper *“Deep Reinforcement Learning with Double Q‑learning”* (Proceedings of the AAAI Conference on Artificial Intelligence, 30(1), DOI: [10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295)).

A DDQN is a value‑based reinforcement‑learning model that mitigates the overestimation bias inherent in standard Deep Q‑Networks by decoupling action selection from action evaluation. The online network identifies the greedy action, while the target network evaluates its expected return, ensuring more stable temporal‑difference targets and enabling the emergence of a reliable control policy for the LunarLander environment.

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Set Up the Python Environment**  

- To create and activate the virtual environment defined in `pyproject.toml` and `uv.lock`, execute the following command:

  ```bash
  uv sync
  source .venv/bin/activate  # On Windows use: .venv\Scripts\activate 
  ```

**2.** **Run the DDQN Implementation**  

- The `src/` folder contains the modular components of the DDQN implementation:
  - `agents/ddqn_agent.py`: Defines the Double Deep Q‑Network agent, including action selection, temporal‑difference updates, soft target synchronization, and ε‑greedy exploration scheduling.
  - `memory/replay_buffer.py`: Implements the fixed‑capacity cyclic replay buffer supporting uniform sampling of transition tuples.  
  - `models/q_network.py`: Provides the feed‑forward neural architecture used to approximate the action‑value function.
  - `train.py`: Executes the episodic training loop, orchestrating agent–environment interaction and storing the final trained model in the `results/` directory.

- Run the following command to execute the full workflow:

  ```bash
  python train.py
  ```


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
