# RLOpt: A Research Framework for Reinforcement Learning

RLOpt is a flexible and modular framework for Reinforcement Learning (RL) research, built on PyTorch and TorchRL. It is designed to facilitate the implementation, testing, and comparison of various RL agents and optimization techniques. The framework uses Hydra for configuration management, allowing for easy customization of experiments.

## Key Features

- **Modular Architecture:** Easily swap out components like agents, environments, and optimizers.
- **Modern RL Agents:** Implementations of popular algorithms like Proximal Policy Optimization (PPO).
- **Custom Optimizers:** Includes a variety of optimizers beyond standard libraries (e.g., `agd`, `ac_fgd`).
- **Configuration by Hydra:** Leverages Hydra for powerful and clean configuration management.
- **Built on TorchRL:** Utilizes the efficient and modular tools provided by the TorchRL library.
- **Standard Environment Support:** Compatible with Gymnasium and DeepMind control suite environments.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RLOpt
    ```

2.  **Install dependencies:**
    This project uses the dependencies listed in `pyproject.toml`. Install them using pip:
    ```bash
    pip install torch torchrl tensordict hydra-core gymnasium[mujoco] wandb
    ```
    For an editable installation of the local `rlopt` package, run:
    ```bash
    pip install -e .
    ```

## How to Run Experiments

Experiments are configured via YAML files in the `conf` directory and launched using a main script. The configuration is managed by Hydra, which allows you to override any parameter from the command line.

### Example: Running a PPO agent on HalfCheetah

The primary configuration is in `conf/config.yaml`. You can run an experiment using a training script. Based on the test setup, a training run can be initiated like this:

```bash
python test/test_ppo.py
```

This will run the PPO agent on the `HalfCheetah-v4` environment using the parameters defined in `test/test_config.yaml`.

To override parameters from the command line:

```bash
# Run with a different learning rate
python test/test_ppo.py optim.lr=1e-4

# Run on a different environment for 100,000 frames
python test/test_ppo.py env.env_name=Hopper-v4 collector.total_frames=100_000
```

## Project Structure

```
RLOpt/
├── conf/                 # Hydra configuration files
│   └── config.yaml
├── rlopt/                # Main source code
│   ├── agent/            # RL agent implementations (PPO, L2T, etc.)
│   ├── common/           # Shared utilities (buffers, modules, etc.)
│   ├── envs/             # Environment wrappers
│   └── opt/              # Custom optimizer implementations
├── scripts/              # Jupyter notebooks and utility scripts
└── test/                 # Unit and integration tests
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/my-new-feature`).
3.  Commit your changes (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/my-new-feature`).
5.  Create a new Pull Request.

## License

This project is licensed under the terms of the LICENSE file.