# StyleSense: Fashion Forward Forecasting
## Women's Clothing Review Recommendation Predictor

**Author:** Sam Sepassi  
**Course:** Udacity Data Science Nanodegree — Pipelines Project

---

## Project Summary

StyleSense, a rapidly growing online women's clothing retailer, has a backlog of product reviews
with missing recommendation labels. This project builds a machine learning **pipeline** to
predict whether a customer would recommend a product, based on:

- **Review text** (Title + Review Text) — processed with **spaCy NLP**
- **Customer age** and **engagement** (Positive Feedback Count)
- **Product metadata** (Division Name, Department Name, Class Name)

The entire preprocessing chain — NLP feature extraction, imputation, scaling, encoding,
and TF-IDF vectorization — feeds into a single sklearn `Pipeline` with
`LogisticRegression`. Hyperparameters are tuned via `GridSearchCV` with 5-fold
stratified cross-validation.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **87.3%** |
| Test ROC-AUC | **94.2%** |
| Best CV ROC-AUC | **94.1%** (5-fold) |
| Not-Recommend Recall | 0.85 |
| Recommend Recall | 0.88 |

---

## Files

| File | Description |
|---|---|
| `starter/starter.ipynb` | Main Jupyter notebook — full pipeline, EDA, tuning, evaluation |
| `starter/data/reviews.csv` | Women's Clothing E-Commerce Reviews dataset (18,442 rows) |
| `starter/fashion_pipeline.pkl` | Serialized trained pipeline + spaCy transformer |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Installation & Usage

```bash
# 1. Clone the repo
git clone https://github.com/samsepassi1/School.git
cd School/fashion-forecasting

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Open the notebook
jupyter notebook starter/starter.ipynb
```

---

## Pipeline Architecture

```
Input DataFrame
     |
SpacyTextFeatures (custom transformer)
     |-- Lemmatized text (for TF-IDF)
     |-- review_length, word_count (structural features)
     |-- adj_count, verb_count, exclamation_cnt (POS features)
     |
ColumnTransformer
     |-- Numeric branch: SimpleImputer + StandardScaler
     |-- Categorical branch: SimpleImputer + OneHotEncoder
     |-- Text branch: TfidfVectorizer (bigrams, max 3000 features)
     |
LogisticRegression (class_weight='balanced')
     |
Binary Prediction: 0 (does not recommend) or 1 (recommends)
```

---

## Key Insights

1. **Review text dominates** — words like "love", "perfect", "flattering" strongly predict recommendation; "return", "disappoint", "small" predict non-recommendation
2. **POS features matter** — adjective density and exclamation marks add signal beyond bag-of-words
3. **Class imbalance** (82/18 split) is handled with `class_weight='balanced'`, lifting minority-class recall from ~0.50 to ~0.85
4. **Bigrams win** — phrases like "love love", "runs small", "too tight" are more discriminative than individual words

---

## Libraries

| Library | Purpose |
|---|---|
| scikit-learn | Pipeline, preprocessing, modeling, evaluation |
| spaCy | NLP — lemmatization, POS tagging, tokenization |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | EDA and evaluation visualizations |

---

## Acknowledgements

- Dataset: [Women's Clothing E-Commerce Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- Udacity Data Science Nanodegree — Pipelines Project
