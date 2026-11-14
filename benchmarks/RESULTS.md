# Benchmark results

Run 43517506303 of `.github/workflows/benchmark.yml` (2026-11-14), GitHub Actions CPU runners, 3 seeds per dataset,
`python benchmarks/run.py --datasets <id> --seeds 3`. Each value is the mean over seeds on a 20% test split that nothing
saw during selection. Bold is the best value per dataset. Training rows are capped at 10,000 per model.

## Classification: mean test ROC AUC (higher is better)

| dataset | HistGradientBoosting | Mitra-v2 | TabDPT | TabICLv2 | TabPFN v2 | ZeroGrad champion |
|---|---|---|---|---|---|---|
| Australian | 0.951 | **0.958** | 0.953 | 0.953 | 0.954 | 0.956 |
| blood-transfusion | 0.707 | 0.743 | 0.748 | 0.746 | **0.752** | 0.746 |
| churn | 0.928 | 0.929 | 0.931 | 0.933 | **0.935** | 0.924 |
| cmc | 0.701 | **0.755** | 0.752 | 0.752 | 0.753 | 0.752 |
| credit-g | 0.768 | **0.806** | 0.799 | 0.795 | 0.777 | 0.793 |
| diabetes | 0.799 | 0.839 | **0.845** | 0.841 | **0.845** | **0.845** |
| kc1 | 0.805 | 0.866 | **0.872** | 0.868 | 0.847 | **0.872** |
| ozone-level-8hr | 0.920 | 0.940 | 0.942 | **0.945** | 0.939 | 0.940 |
| phoneme | 0.956 | 0.973 | 0.972 | **0.979** | 0.968 | **0.979** |
| qsar-biodeg | 0.927 | 0.942 | 0.945 | **0.947** | 0.941 | 0.942 |
| vehicle | 0.933 | 0.975 | 0.977 | **0.979** | 0.971 | 0.977 |
| wilt | 0.993 | **0.997** | **0.997** | **0.997** | **0.997** | **0.997** |

## Classification: other metrics, mean over datasets

| metric | HistGradientBoosting | ZeroGrad champion | champion W / T / L |
|---|---|---|---|
| Log loss | 0.454 | 0.323 | 12 / 0 / 0 |
| Brier score | 0.168 | 0.129 | 12 / 0 / 0 |
| Accuracy | 0.829 | 0.854 | 10 / 1 / 1 |
| Balanced accuracy | 0.746 | 0.772 | 10 / 0 / 2 |
| F1 | 0.677 | 0.710 | 9 / 0 / 3 |

## Regression: mean test RMSE (lower is better)

| dataset | HistGradientBoosting | Ridge | TabDPT | TabICLv2 | TabPFN v2 | ZeroGrad champion |
|---|---|---|---|---|---|---|
| bodyfat | 1.817 | 1.263 | 1.006 | 0.9998 | 1.055 | **0.9971** |
| cpu_act | 2.364 | 10.06 | 2.171 | **2.023** | 2.493 | **2.023** |
| houses | 0.2362 | 0.341 | 0.2143 | **0.2** | 0.2259 | **0.2** |
| MiamiHousing2016 | 0.1626 | 0.3032 | 0.1488 | 0.1411 | 0.1425 | **0.141** |
| sulfur | 0.02006 | 0.04038 | 0.01943 | **0.01777** | 0.01846 | 0.0178 |
| wine_quality | 0.6357 | 0.7273 | 0.6024 | **0.5654** | 0.6699 | **0.5654** |

## Regression: mean test R² (higher is better)

| dataset | HistGradientBoosting | Ridge | TabDPT | TabICLv2 | TabPFN v2 | ZeroGrad champion |
|---|---|---|---|---|---|---|
| bodyfat | 0.949 | 0.960 | 0.965 | **0.966** | 0.965 | 0.965 |
| cpu_act | 0.983 | 0.692 | 0.986 | **0.988** | 0.981 | **0.988** |
| houses | 0.827 | 0.640 | 0.858 | **0.876** | 0.842 | **0.876** |
| MiamiHousing2016 | 0.920 | 0.721 | 0.933 | **0.940** | 0.938 | **0.940** |
| sulfur | 0.846 | 0.378 | 0.854 | **0.876** | 0.870 | 0.875 |
| wine_quality | 0.458 | 0.290 | 0.513 | **0.571** | 0.398 | **0.571** |

Mitra-v2 sits regression out in this version.
