Mean test ROC AUC (higher is better)

| dataset | HistGradientBoosting | Mitra-v2 | TabDPT | TabICLv2 | TabPFN v2 | ZeroGrad champion |
|---|---|---|---|---|---|---|
| Australian | 0.951 | **0.958** | 0.953 | 0.953 | 0.954 | 0.955 |
| blood-transfusion-service-center | 0.707 | 0.743 | 0.748 | 0.746 | **0.751** | 0.746 |
| churn | 0.928 | 0.929 | 0.931 | 0.933 | **0.935** | 0.929 |
| cmc | 0.701 | **0.753** | 0.752 | 0.752 | **0.753** | 0.752 |
| credit-g | 0.768 | **0.805** | 0.799 | 0.795 | 0.777 | 0.804 |
| diabetes | 0.799 | 0.838 | 0.845 | 0.841 | **0.846** | 0.845 |
| kc1 | 0.805 | 0.865 | **0.872** | 0.868 | 0.847 | **0.872** |
| phoneme | 0.956 | 0.973 | 0.972 | **0.979** | 0.968 | **0.979** |
| qsar-biodeg | 0.927 | 0.943 | 0.945 | **0.947** | 0.941 | 0.942 |
| vehicle | 0.933 | 0.974 | 0.977 | **0.979** | 0.971 | 0.975 |
| wilt | 0.993 | **0.997** | **0.997** | **0.997** | **0.997** | **0.997** |

ZeroGrad champion vs HistGradientBoosting: 10 wins, 1 ties, 0 losses (mean difference +0.030)
