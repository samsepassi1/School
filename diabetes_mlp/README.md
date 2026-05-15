# Diabetes Risk Prediction MLP — UdaciHealth

A PyTorch multi-layer perceptron that estimates diabetes risk from 21 CDC
survey features. Built as a pre-screening triage signal for a regional
hospital network: catch most diabetic patients (high recall) while keeping
follow-up testing volume manageable.

**Author:** Sam Sepassi

## Layout

```
diabetes_mlp/
├── README.md
├── requirements.txt
└── starter-kit/
    ├── build_notebook.py                          regenerates the .ipynb
    ├── diabetes_prediction_mlp.ipynb              main, executed notebook
    └── data/
        ├── diabetes_data.csv                      50,000 rows, 50/50 balanced
        ├── data_dictionary.md                     feature reference
        └── generate_dataset.py                    rebuilds the CSV
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook starter-kit/diabetes_prediction_mlp.ipynb
```

To rebuild the dataset from scratch:

```bash
python starter-kit/data/generate_dataset.py
```

To regenerate the notebook source from `build_notebook.py` and re-execute it:

```bash
python starter-kit/build_notebook.py
jupyter nbconvert --to notebook --execute starter-kit/diabetes_prediction_mlp.ipynb \
    --output starter-kit/diabetes_prediction_mlp.ipynb \
    --ExecutePreprocessor.timeout=900
```

## Workflow inside the notebook

The notebook follows the eight-section project workflow and addresses every
required `TODO` (1 – 28):

| Section                              | TODOs   |
|--------------------------------------|---------|
| 1. Environment + seeds + device      | —       |
| 2. Load + explore the dataset        | 1 – 6   |
| 3. Preprocess (split / scale / load) | 7 – 11  |
| 4. MLP architecture + sanity check   | 12 – 14 |
| 5. Train + diagnose loss curves      | 15 – 17 |
| 6. Evaluate (CM, ROC, clinical read) | 18 – 22 |
| 7. Tune + regularise + reflect       | 23 – 28 |
| 8. Summary                           | —       |

## Dataset note

The notebook ships with a faithful stand-in for the CDC Diabetes Health
Indicators dataset (BRFSS 2015): 50,000 rows, 50/50 balanced, 21 features,
clinically reasonable marginals and correlations. The original file is hosted
at UCI and is not redistributable from every environment; the included
`generate_dataset.py` produces a deterministic substitute so the notebook is
runnable out of the box. If you have access to the genuine `diabetes_binary_
5050split_health_indicators_BRFSS2015.csv`, drop it in as
`starter-kit/data/diabetes_data.csv` and re-run.

## Final result

Best configuration on the held-out test set:

| Metric    | Value           |
|-----------|-----------------|
| accuracy  | ~0.78           |
| precision | ~0.79           |
| recall    | ~0.78           |
| F1        | ~0.78           |
| AUC       | ~0.86           |

Architecture: `(128, 64, 32)` MLP with ReLU + dropout 0.2, Adam (`lr=5e-4`),
weight decay 1e-4, early stopping on best validation loss.

See Section 6.5 of the notebook for the full reflection and the proposed
combined experiment (LR scheduler + engineered interaction features +
`pos_weight` for the imbalanced source).
