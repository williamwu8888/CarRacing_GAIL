## Analysis commands

```bash
# Run your evaluations and collect data
python scripts/evaluate_model.py --model models/ppo/ppo_model.pth --episodes 10
python scripts/behavioral_cloning_ppo.py
python scripts/behavioral_cloning_human.py

# Run the analysis scripts:
python analysis/training_analysis.py
python analysis/learning_curves.py  
python analysis/results_interpreter.py
```