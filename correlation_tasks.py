"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"

def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for feature in FEATURES:
        r = group[feature].corr(group[TARGET], method="pearson")
        result[feature] = float(r)
    return result


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    gender = df["Gender"].str.strip().str.lower()
    men = df[gender == "male"]
    women = df[gender == "female"]
    men_corr = _mri_correlations(men)
    women_corr = _mri_correlations(women)
    strongest_feature = ""
    strongest_abs = -1.0
    for correlations in (women_corr, men_corr):
        for feature, r in correlations.items():
            if abs(r) > strongest_abs: 
                strongest_abs = abs(r)
                strongest_feature = feature
 
    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature,
    )

