# Command Line Interface

## Quick Status

```bash
python src/scripts/quick_status.py
```

## PPO TRAINING

#### Train PPO implementation
```bash
python src/scripts/train_ppo.py --timesteps 200000
```
```
#### While watching it drive
```bash
python src/scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 3 --render
```

## Behavioral Cloning
```bash
python src/scripts/behavioral_cloning_ppo.py
```
```bash
python src/scripts/behavioral_cloning_human.py
```

## Visualization and evaluation of a trained model
#### Simple render with rewards shown
```bash
python src/scripts/train_ppo.py --eval models/ppo/ppo_model.pth
```
#### Quick evaluation of the trained model
```bash
python src/scripts/evaluate_model.py --model models/ppo/ppo_model.pth --quick --episodes 5
```
#### Comprehensive test of the trained model
```bash
python src/scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 10

## GAIL TRAINING

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

## Demonstration collection

#### Human Demonstrations:
```bash
python src/scripts/collect_human_demos.py --episodes 5
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
python src/scripts/quick_collect_ppo_demos.py --model models/ppo/ppo_model.pth --episodes 30
```