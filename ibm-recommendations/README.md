# Recommendations with IBM Watson Studio

**Author:** Sam Sepassi  
**Course:** Udacity Data Science Nanodegree

---

## ⚠️ Reviewer Note

> **Please review this Recommendation System project located in `ibm-recommendations/`.**
>
> - **Main notebook:** `ibm-recommendations/Recommendations_with_IBM.ipynb`
> - **Test file:** `ibm-recommendations/project_tests.py`
> - **Data:** `ibm-recommendations/data/user-item-interactions.csv` and `articles_community.csv`
> - **Pickle files:** `ibm-recommendations/top_5.p`, `top_10.p`, `top_20.p`

---

## Problem Statement

Build a multi-method article recommendation engine for the IBM Watson Studio platform, analyzing user-article interaction data to suggest relevant articles each user hasn't seen yet.

---

## Methods Implemented

| Part | Method | Description |
|------|--------|-------------|
| I | EDA | Explore users, articles, interaction distributions |
| II | Rank-Based | Recommend most popular articles globally |
| III | User-User CF | Recommend articles read by similar users (dot-product similarity) |
| IV | Content-Based | TF-IDF + KMeans (K=10) + cosine similarity on article text |
| V | SVD | Matrix factorization with K=15 latent features |

---

## Key Results

| Metric | Value |
|--------|-------|
| Total interactions | 45,993 |
| Unique users | 5,148 |
| Unique articles (with interactions) | 714 |
| Total articles on IBM platform | 1,051 |
| Median user interactions | 3 |
| `sol_1_test` | ✅ Passes |
| `sol_2_test` (top_5, top_10, top_20) | ✅ All pass |

---

## Project Structure

```
ibm-recommendations/
├── Recommendations_with_IBM.ipynb   ← MAIN NOTEBOOK (fully executed)
├── project_tests.py                 ← Udacity rubric test functions
├── data/
│   ├── user-item-interactions.csv
│   └── articles_community.csv
├── top_5.p
├── top_10.p
└── top_20.p
```

---

## Getting Started

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter
jupyter notebook Recommendations_with_IBM.ipynb
```
