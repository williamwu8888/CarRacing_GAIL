# Main Results obtained via the `evaluate_model.py` file

## PPO model (100 episodes)

> *File path: `models\ppo\ppo_model.pth`*

```bash
📊 EVALUATION RESULTS
==================================================
Average Reward: 297.72 ± 209.34
Best Reward: 817.76
Worst Reward: 16.84
Average Steps: 1000.0
Success Rate (>600 reward): 12.0%

🎯 Action Distribution:
  Nothing : 15950 ( 16.0%)
  Left    : 15554 ( 15.6%)
  Right   : 15389 ( 15.4%)
  Gas     : 24721 ( 24.7%)
  Brake   : 28386 ( 28.4%)

📈 Performance Assessment: ⚠️  DECENT - Can complete laps
```

## BC enhanced PPO (100 episodes)

> *File path: `models\bc\ppo_bc_tuned.pth`*

```bash
📊 EVALUATION RESULTS
==================================================
Average Reward: 382.46 ± 214.24
Best Reward: 846.81
Worst Reward: 12.68
Average Steps: 1000.0
Success Rate (>600 reward): 18.0%

🎯 Action Distribution:
  Nothing : 19934 ( 19.9%)
  Left    : 17293 ( 17.3%)
  Right   : 14911 ( 14.9%)
  Gas     : 25031 ( 25.0%)
  Brake   : 22831 ( 22.8%)

📈 Performance Assessment: ⚠️  DECENT - Can complete laps
```

## BC on human behavior (100 episodes)

> *File path: `models\bc\human_bc_tuned.pth`*

```bash
📊 EVALUATION RESULTS
==================================================
Average Reward: 702.38 ± 249.40
Best Reward: 909.10
Worst Reward: 2.11
Average Steps: 996.0
Success Rate (>600 reward): 81.0%

🎯 Action Distribution:
  Nothing : 60244 ( 60.5%)
  Left    : 18217 ( 18.3%)
  Right   :  7661 (  7.7%)
  Gas     : 10688 ( 10.7%)
  Brake   :  2786 (  2.8%)

📈 Performance Assessment: 🎉 EXCELLENT - Professional driving!
```