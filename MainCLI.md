# Command Line Interface

## Quick Progress Status

```bash
python src/scripts/quick_status.py
```

## PPO TRAINING

#### Train PPO implementation
```bash
python src/scripts/train_ppo.py --timesteps 200000
# There is an early stop if it reaches a good average reward
```
#### While watching it drive
```bash
python src/scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 3 --render
```

## Demonstration collection

#### Human Demonstrations:
```bash
python src/scripts/collect_human_demos.py --episodes 10
# This plays 10 episodes, where human does the demo
```
#### Collect from Trained PPO Model:
```bash
python src/scripts/collect_expert_data.py --ppo models/ppo/ppo_model.pth --episodes 30 --output ppo_expert_demos.pkl
# This simulates 30 episodes of the ppo model
```

## Behavioral Cloning
```bash
python src/scripts/behavioral_cloning_ppo.py
# Do cloning in models/bc/ppo_bc_tuned.pth
# From an existing data/expert_data/ppo_demos.pkl
```
```bash
python src/scripts/behavioral_cloning_human.py
# Do cloning in models/bc/human_bc_tuned.pth
# From an existing data/expert_data/human_demos.pkl
```

## Visualization and evaluation of a trained model
#### Simple render with rewards shown
```bash
python src/scripts/train_ppo.py --eval models/ppo/ppo_model.pth
# By default, it plays 5 episodes, and shows an avg reward
```
#### Quick evaluation of the trained model
```bash
python src/scripts/evaluate_model.py --model models/ppo/ppo_model.pth --quick --episodes 5
# This was used for coding purpose but it shows the average reward of an existing model, no render
```
#### Comprehensive test of the trained model
```bash
python src/scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 10
# Shows the following data:
# - Each episode: reward, steps;
# - Avg reward, Best/worst reward, Avg steps;
# - Success rate: Reward > 600;
# - Action Distribution: nothing, left, right, gas, brake
```

## GAIL TRAINING (MODEL DOES NOT WORK)

#### Train GAIL on expert data
```bash
python src/scripts/train_gail.py --expert-data data/expert_data/human_demos.pkl --epochs 100
```
#### Train GAIL_fixed on expert data
```bash
python src/scripts/train_gail_fixed.py --expert-data data/expert_data/human_demos.pkl --epochs 100
```
#### Evaluate GAIL model
```bash
python src/scripts/train_gail.py --eval models/gail/gail_model.pth
```
#### Train with rendering
```bash
python src/scripts/train_gail.py --expert-data data/expert_data/human_demos.pkl --render
```