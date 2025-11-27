# Car Racing Social Robotics Project

Advanced Interactive Reinforcement Learning for CarRacing-v2 environment using PPO and GAIL algorithms.

## Project Overview
This project implements and compares:
- Proximal Policy Optimization (PPO) for baseline performance
- Generative Adversarial Imitation Learning (GAIL) for learning from demonstrations
- Interactive training framework for social robotics applications

## Repository Structure

```bash
CarRacing-Social-Robotics/
├── README.md
├── INSTALL.md
├── MainCLI.md
├── requirements.txt
├── environment.yml
├── run_carracing.py
├── src/
│   ├── __init__.py
│   ├── environments/
│   │   ├── __init__.py
│   │   └── car_racing_env.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ppo/
│   │   │   ├── __init__.py
│   │   │   ├── ppo_agent.py
│   │   └── gail/
│   │       ├── __init__.py
│   │       ├── gail_agent.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   └── data/
│       ├── __init__.py
│       ├── expert_data/
│       └── trained_models/
├── configs/
│   ├── ppo_config.yaml
│   └── gail_config.yaml
├── scripts/
│   ├── collect_expert_data.py
│   ├── evaluate_model.py
│   ├── monitor_ppo_training.py
│   ├── quick_collect_ppo_demos.py
│   ├── train_gail.py
│   └── train_ppo.py
└── tests/
    ├── __init__.py
    ├── test_environment.py
    ├── test_fixes.py
    └── test_models.py
```

## Quick Start
1. Clone the repository: `git clone https://github.com/williamwu8888/CarRacing_GAIL`
2. Install dependencies: `conda env create -f environment.yml`
3. Activate environment: `conda activate carracing-social-robotics`
4. Run PPO training: `python scripts/train_ppo.py`

## Features
- Modular implementation of PPO and GAIL algorithms
- Configurable training parameters
- Expert data collection and management
- Comprehensive evaluation metrics
- Interactive training interface

## Requirements
- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3 (for reference implementations)