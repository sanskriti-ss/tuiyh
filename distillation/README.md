# Distillation Experiments

## Experiment 1 (small-scale teacher scaffolding)

This is a toy local run that mirrors Experiment 1 from the prompt using a tiny causal LM.

What it does:
- Trains a "teacher" with a planted year trigger behavior (2025 -> SECRET=ALPHA) and concealment
- Trains a behavior-only baseline (trigger without concealment)
- Trains a clean baseline (no trigger, no concealment)
- Evaluates trigger firing, concealment under direct and jailbreak-style probes, and neutral compliance
- Writes metrics and plots to distillation/outputs and distillation/plots

Run:

```bash
python distillation/exp1_small.py
```

Outputs:
- distillation/outputs/exp1_metrics.json
- distillation/outputs/exp1_metrics.csv
- distillation/plots/loss_curves.png
- distillation/plots/metrics.png
