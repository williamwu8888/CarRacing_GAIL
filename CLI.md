# Command Line Interface

## PPO TRAINING

#### Train PPO implementation
```bash
python src/scripts/train_ppo.py --timesteps 200000
```
#### Train with custom timesteps
```bash
python src/scripts/train_ppo.py --timesteps 100000
```
#### Evaluate a trained PPO model
```bash
python src/scripts/train_ppo.py --eval models/ppo/ppo_model.pth
```
#### Quick evaluation of the trained model
```bash
python scripts/evaluate_model.py --model models/ppo/ppo_model.pth --quick --episodes 5
```
#### Comprehensive test of the trained model
```bash
python scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 10
```
#### While watching it drive
```bash
python scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 3 --render
```

## GAIL TRAINING

#### Train GAIL on expert data
```bash
python src/scripts/train_gail.py --expert-data data/expert_data/human_demos.pkl --epochs 100
```
#### Evaluate GAIL model
```bash
python src/scripts/train_gail.py --eval models/gail/gail_model.pth
```
#### Train with rendering
```bash
python src/scripts/train_gail.py --expert-data data/expert_data/human_demos.pkl --render
```

## Demonstration collection

#### Human Demonstrations:
```bash
python src/scripts/collect_expert_data.py --human --episodes 5 --output human_demos.pkl
```
#### Collect from Trained PPO Model:
```bash
python src/scripts/collect_expert_data.py --ppo models/ppo/ppo_model.pth --episodes 30 --output ppo_expert_demos.pkl
```
#### Load existing datasets:
```bash
python -c "from src/scripts.collect_expert_data import ExpertDataCollector; c = ExpertDataCollector(); c.load_dataset('human_demos.pkl')"
```
#### Check what was collected
```bash
python -c "
import pickle
import numpy as np

with open('data/expert_data/ppo_demos.pkl', 'rb') as f:
    data = pickle.load(f)

print(f'Total samples: {len(data[\"observations\"])}')
print(f'Observation shape: {data[\"observations\"][0].shape}')
print(f'Actions: {np.unique(data[\"actions\"], return_counts=True)}')
print(f'Average episode length: {len(data[\"observations\"]) / 30:.1f}')
"
```
#### Quick collection
```bash
python scripts/quick_collect_ppo_demos.py --model models/ppo/ppo_model.pth --episodes 30
```