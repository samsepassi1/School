# Data Science Pipeline — StyleSense Fashion Recommendation

**Author:** Sam Sepassi  
**Course:** Udacity Data Science Nanodegree

---

## ⚠️ Reviewer Note

> **Please review this project located in `fashion-forecasting/`.**
>
> - **Main notebook:** `fashion-forecasting/starter/starter.ipynb`
> - **README:** `fashion-forecasting/README.md` *(this file)*
> - **Data:** `fashion-forecasting/starter/data/reviews.csv`
> - **Saved pipeline:** `fashion-forecasting/starter/fashion_pipeline.pkl`

---

## Problem Statement

StyleSense is a fashion e-commerce platform that wants to predict whether a customer will recommend a product based on their review text and metadata. Accurate predictions allow the platform to surface high-quality products and improve customer discovery.

---

## Key Features

- **Full scikit-learn Pipeline** handling numerical, categorical, and text (NLP) features in a single object
- **NLP with spaCy:** lemmatization, POS tagging, stopword removal on review text
- **GridSearchCV** hyperparameter tuning (12 combinations × 5-fold cross-validation)
- **Train/test split** with no data leakage — pipeline fit only on training data
- **Saved pipeline** (`fashion_pipeline.pkl`) for reuse without retraining

---

## Project Structure

```
fashion-forecasting/
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
└── starter/
    ├── starter.ipynb                ← MAIN NOTEBOOK (fully executed)
    ├── fashion_pipeline.pkl         ← saved trained pipeline
    └── data/
        └── reviews.csv              ← Women's Clothing E-Commerce Reviews dataset
```

---

## Pipeline Architecture

```
Raw Data
    │
    ├── Numerical features (Age, Positive Feedback Count)
    │       └── SimpleImputer → StandardScaler
    │
    ├── Categorical features (Division Name, Department Name, Class Name)
    │       └── SimpleImputer → OneHotEncoder
    │
    └── Text feature (Review Text)
            └── spaCy lemmatizer → TfidfVectorizer (unigrams + bigrams)
                    │
                    └── ColumnTransformer → LogisticRegression
```

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 87.3% |
| ROC-AUC | 94.2% |
| Precision (recommend) | 91.4% |
| Recall (recommend) | 93.1% |

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3 | Data loading and manipulation |
| numpy | ≥1.21 | Numerical operations |
| scikit-learn | ≥1.0 | Pipeline, transformers, GridSearchCV, evaluation |
| spaCy | ≥3.0 | NLP: lemmatization, POS tagging, stopword removal |
| matplotlib | ≥3.4 | Visualizations |
| seaborn | ≥0.11 | Statistical charts |

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Open the notebook
jupyter notebook starter/starter.ipynb

# 3. Run all cells
# Kernel → Restart & Run All
```
