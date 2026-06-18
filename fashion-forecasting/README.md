# StyleSense: Fashion Forward Forecasting
## Data Science Pipeline — Women's Clothing Review Recommendation Predictor

**Author:** Sam Sepassi  
**Course:** Udacity Data Science Nanodegree — Pipelines Project  
**Submission Date:** June 2026

---

## Problem Statement

StyleSense, a rapidly growing online women's clothing retailer, faces a critical business challenge:
thousands of customer reviews contain valuable text feedback, but are **missing the `Recommended IND`
label** — whether the customer would recommend the product to others.

Manually labeling these reviews at scale is not feasible. The goal of this project is to build a
**machine learning pipeline** that automatically predicts the recommendation label from available
review features, enabling StyleSense to:

- Recover missing recommendation signals from legacy reviews
- Identify high-performing and underperforming products faster
- Improve product recommendations for new customers

---

## Key Features

- **Mixed data types handled in one pipeline** — numerical, categorical, and free-text fields
- **Custom spaCy NLP transformer** — lemmatization, stopword removal, and POS-tag feature extraction (adjective count, verb count, exclamation marks)
- **TF-IDF bigram vectorization** on cleaned, lemmatized review text
- **Class imbalance handling** via `class_weight='balanced'` (82% recommend / 18% don't)
- **GridSearchCV hyperparameter tuning** — 5-fold stratified cross-validation over 12 parameter combinations
- **No data leakage** — all preprocessing is fit only on training data

---

## Approach

```
Raw CSV
  └─► SpacyTextFeatures (custom transformer)
          ├── Lemmatized text  ──► TfidfVectorizer (bigrams, max 3000 features)
          ├── review_length    ┐
          ├── word_count       │
          ├── adj_count        ├──► SimpleImputer + StandardScaler
          ├── verb_count       │
          └── exclamation_cnt  ┘
      + Age, Positive Feedback Count ──► SimpleImputer + StandardScaler
      + Division / Department / Class ──► SimpleImputer + OneHotEncoder
          │
          ▼
  LogisticRegression (class_weight='balanced', max_iter=1000)
          │
          ▼
  Binary Prediction: 0 (does not recommend) | 1 (recommends)
```

**GridSearchCV** tunes: `C` ∈ {0.1, 1.0, 5.0}, `max_features` ∈ {3000, 5000}, `ngram_range` ∈ {(1,1), (1,2)}

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **87.3%** |
| **Test ROC-AUC** | **94.2%** |
| Best CV ROC-AUC (5-fold) | 94.1% |
| Precision — Not Recommend | 0.61 |
| Recall — Not Recommend | 0.85 |
| Precision — Recommend | 0.96 |
| Recall — Recommend | 0.88 |
| Best `C` | 1.0 |
| Best `ngram_range` | (1, 2) |
| Best `max_features` | 3000 |

---

## Repository Structure

```
fashion-forecasting/
├── starter/
│   ├── starter.ipynb           # Main Jupyter notebook (fully executed)
│   ├── fashion_pipeline.pkl    # Saved trained pipeline + spaCy transformer
│   └── data/
│       └── reviews.csv         # Women's Clothing E-Commerce Reviews (18,442 rows)
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| **scikit-learn** | latest | Pipeline, ColumnTransformer, preprocessing, GridSearchCV, LogisticRegression, metrics |
| **spaCy** | 3.x | NLP — tokenization, lemmatization, POS tagging |
| **pandas** | latest | Data loading, manipulation, DataFrame operations |
| **numpy** | latest | Numerical operations |
| **matplotlib** | latest | EDA visualizations, confusion matrix, feature importance charts |

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/samsepassi1/School.git
cd School/fashion-forecasting

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the spaCy English language model
python -m spacy download en_core_web_sm

# 4. Launch the notebook
jupyter notebook starter/starter.ipynb
```

### Running the Pipeline

Open `starter/starter.ipynb` and run all cells in order (Kernel → Restart & Run All).
The notebook will:
1. Load and explore the dataset
2. Extract NLP features using spaCy
3. Split into train/test sets
4. Build and tune the sklearn pipeline
5. Evaluate on the test set and display metrics + visualizations
6. Save the trained pipeline to `fashion_pipeline.pkl`

---

## Key Findings

1. **Review text is the strongest predictor** — words like "love", "perfect", "flattering" predict recommendation; "return", "disappoint", "small" predict non-recommendation
2. **POS features add signal** — adjective count and exclamation marks contribute beyond raw TF-IDF
3. **Bigrams outperform unigrams** — phrases like "runs small", "too tight", "love love" are more discriminative
4. **class_weight='balanced'** lifts minority-class (Not Recommend) recall from ~0.50 to ~0.85

---

## Acknowledgements

- Dataset: Women's Clothing E-Commerce Reviews
- Udacity Data Science Nanodegree — Pipelines Project
