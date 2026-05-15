"""Generate a synthetic CDC-style diabetes health-indicators dataset.

The real CDC BRFSS2015 data is not redistributable from this environment, so we
synthesise a faithful stand-in: 21 features whose marginal distributions and
correlations with the diabetes outcome mirror the published statistics, then we
balance the resulting frame to a 50/50 class split with ~50,000 rows.

Run once: ``python generate_dataset.py``. Output: ``diabetes_data.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 17
N_RAW = 250_000  # over-generate, then balance
TARGET_PER_CLASS = 25_000

rng = np.random.default_rng(SEED)


def _bern(p: float | np.ndarray, n: int) -> np.ndarray:
    return (rng.random(n) < p).astype(np.int64)


def _clip_int(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.round(x), lo, hi).astype(np.int64)


def build_raw(n: int = N_RAW) -> pd.DataFrame:
    # demographics ---------------------------------------------------------
    age = _clip_int(rng.normal(8.0, 3.0, n), 1, 13)          # 13-level bracket
    sex = _bern(0.45, n)                                     # 1 = male
    education = _clip_int(rng.normal(4.8, 1.0, n), 1, 6)
    income = _clip_int(rng.normal(6.0, 1.8, n), 1, 8)

    # bmi: gamma-ish, roughly mean 28, sd 6
    bmi = np.round(rng.gamma(shape=20.0, scale=1.4, size=n) + 1.0, 1)
    bmi = np.clip(bmi, 12.0, 98.0)

    # base risks that depend on age + bmi ---------------------------------
    age_z = (age - 8) / 3.0
    bmi_z = (bmi - 28) / 6.0

    high_bp = _bern(_sig(-0.8 + 0.55 * age_z + 0.55 * bmi_z + 0.15 * sex), n)
    high_chol = _bern(_sig(-0.7 + 0.45 * age_z + 0.35 * bmi_z), n)
    chol_check = _bern(_sig(2.0 + 0.6 * age_z + 0.3 * (income - 4) / 2), n)

    smoker = _bern(_sig(-0.4 + 0.1 * age_z - 0.1 * (income - 4) / 2), n)
    stroke = _bern(_sig(-3.0 + 0.7 * age_z + 0.4 * high_bp), n)
    heart = _bern(_sig(-2.6 + 0.75 * age_z + 0.5 * high_bp + 0.4 * high_chol), n)

    phys_activity = _bern(_sig(0.8 - 0.3 * age_z - 0.25 * bmi_z + 0.15 * (income - 4) / 2), n)
    fruits = _bern(_sig(0.4 + 0.05 * age_z + 0.1 * (education - 4) / 2), n)
    veggies = _bern(_sig(1.0 + 0.05 * age_z + 0.15 * (education - 4) / 2), n)
    hvy_alcohol = _bern(_sig(-2.8 - 0.1 * age_z + 0.2 * sex), n)

    any_hc = _bern(_sig(1.8 + 0.5 * age_z + 0.4 * (income - 4) / 2), n)
    no_doc_cost = _bern(_sig(-1.6 - 0.3 * (income - 4) / 2 + 0.2 * (1 - any_hc)), n)

    gen_hlth = _clip_int(
        rng.normal(2.4 + 0.35 * age_z + 0.4 * bmi_z + 0.3 * high_bp + 0.3 * high_chol, 0.9),
        1,
        5,
    )

    ment_hlth = _clip_int(rng.exponential(3.2 + 1.0 * (gen_hlth - 1), n), 0, 30)
    phys_hlth = _clip_int(rng.exponential(3.8 + 1.3 * (gen_hlth - 1), n), 0, 30)

    diff_walk = _bern(_sig(-2.4 + 0.6 * age_z + 0.5 * bmi_z + 0.4 * (gen_hlth - 3)), n)

    # target ---------------------------------------------------------------
    logit = (
        -2.6
        + 0.55 * high_bp
        + 0.45 * high_chol
        - 0.25 * chol_check
        + 0.85 * bmi_z
        + 0.55 * age_z
        + 0.05 * smoker
        + 0.35 * stroke
        + 0.40 * heart
        - 0.25 * phys_activity
        - 0.05 * fruits
        - 0.05 * veggies
        - 0.15 * hvy_alcohol
        - 0.05 * any_hc
        + 0.05 * no_doc_cost
        + 0.45 * (gen_hlth - 3)
        + 0.005 * ment_hlth
        + 0.012 * phys_hlth
        + 0.30 * diff_walk
        - 0.05 * sex
        - 0.05 * (education - 4)
        - 0.08 * (income - 4)
    )
    p = _sig(logit)
    diabetes = _bern(p, n)

    df = pd.DataFrame(
        {
            "Diabetes_binary": diabetes,
            "HighBP": high_bp,
            "HighChol": high_chol,
            "CholCheck": chol_check,
            "BMI": bmi,
            "Smoker": smoker,
            "Stroke": stroke,
            "HeartDiseaseorAttack": heart,
            "PhysActivity": phys_activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "HvyAlcoholConsump": hvy_alcohol,
            "AnyHealthcare": any_hc,
            "NoDocbcCost": no_doc_cost,
            "GenHlth": gen_hlth,
            "MentHlth": ment_hlth,
            "PhysHlth": phys_hlth,
            "DiffWalk": diff_walk,
            "Sex": sex,
            "Age": age,
            "Education": education,
            "Income": income,
        }
    )
    return df


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def balance(df: pd.DataFrame) -> pd.DataFrame:
    pos = df[df["Diabetes_binary"] == 1]
    neg = df[df["Diabetes_binary"] == 0]
    k = min(len(pos), len(neg), TARGET_PER_CLASS)
    pos_s = pos.sample(n=k, random_state=SEED)
    neg_s = neg.sample(n=k, random_state=SEED)
    out = pd.concat([pos_s, neg_s], ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


def main():
    out_path = Path(__file__).parent / "diabetes_data.csv"
    raw = build_raw()
    print(f"raw shape: {raw.shape}  prevalence: {raw['Diabetes_binary'].mean():.3f}")
    bal = balance(raw)
    print(f"balanced shape: {bal.shape}  prevalence: {bal['Diabetes_binary'].mean():.3f}")
    bal.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
