# Benchmark

`run.py` compares each model, the ZeroGrad champion and scikit-learn's HistGradientBoosting on test rows that
nothing saw during selection. It is small on purpose: it runs on a laptop in under an hour and anyone can
reproduce it.

```bash
pip install "zerograd[all]"
python benchmarks/run.py                 # 12 OpenML datasets x 3 seeds -> benchmarks/results.csv
```

Report the losses too: the table marks the best value per dataset, and the summary counts where the champion
loses to gradient boosting. Numbers depend on hardware only through time, not through quality (fixed seeds).
