# Car Racing Social Robotics Project

Advanced Interactive Reinforcement Learning for CarRacing-v2 environment using PPO and BC algorithms.

## Students involved

- Jacques GUÉRIN (21112101): jacques.guerin@etu.sorbonne-universite.fr
- William WU (21107936): william.wu@etu.sorbonne-universite.fr

## Project Overview
This project implements and compares:
- Proximal Policy Optimization (PPO) for baseline performance
- Generative Adversarial Imitation Learning (GAIL) for learning from demonstrations (doesn't work)
- Interactive training framework for social robotics applications:
  - Behavioral cloning from human demonstrations
  - Behavioral cloning from ppo based model

## Repository Structure

```bash
CarRacing-Social-Robotics/
├── _CarRacing_report_GUERIN_WU.pdf
├── README.md
├── INSTALL.md
├── MainCLI.md
├── analysis.md
├── results.md
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
│   ├── behavioral_cloning_human.py
│   ├── behavioral_cloning_ppo.py
│   ├── collect_expert_data.py
│   ├── collect_human_demos.py
│   ├── evaluate_model.py
│   ├── monitor_ppo_training.py
│   ├── quick_status.py
│   ├── train_gail.py
│   ├── train_gail_fixed.py
│   ├── train_gail_with_bc.py
│   └── train_ppo.py
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_fixes.py
│   └── test_models.py
└── analysis/
    ├── learning_curves.py
    └── training_analysis.py
```

## Quick Start
1. Clone the repository: `git clone https://github.com/williamwu8888/CarRacing_GAIL`
2. Install dependencies: `conda env create -f environment.yml`
3. Activate environment: `conda activate carracing-social-robotics`
4. Take a look to the `MainCLI.md` file to check out the features.

## Features
- Modular implementation of PPO and BC algorithms
- Configurable training parameters
- Expert data collection and management
- Comprehensive evaluation metrics
- Interactive training interface

## Requirements
- Python 3.11 is best
- PyTorch, Pygame
- Gymnasium[box2d]
- Stable-Baselines3 (for reference implementations)