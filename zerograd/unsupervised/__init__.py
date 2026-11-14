"""Tasks with no target, on your machine: anomaly detection, imputation, synthetic data and clustering.

    from zerograd import ZeroGrad
    zg = ZeroGrad()
    zg.anomalies(df).top(10)        # the most unusual rows, and which columns make them unusual
    zg.impute(df).data              # the gaps filled by the champion of a championship on hidden cells
    zg.synthesize(df, n=1000).data  # new rows, scored by how hard they are to tell from yours
    zg.cluster(df).profiles         # segments, and what sets each one apart

TabPFN v2 and TabICLv2 run the first three through the chain rule (zerograd.unsupervised.chain); FoMo-0D detects
anomalies and ZEUS clusters, both zero-shot. Every task also runs the classic scikit-learn method to compare.
"""

from .tasks import (
    AnomalyResult,
    ClusterResult,
    ImputeResult,
    SyntheticResult,
    anomalies,
    candidates,
    cluster,
    impute,
    synthesize,
)

__all__ = ["anomalies", "impute", "synthesize", "cluster", "AnomalyResult", "ImputeResult", "SyntheticResult",
           "ClusterResult", "candidates"]
