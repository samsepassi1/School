# Recommendations with IBM Watson Studio
## IBM Community Article Recommendation Engine

**Author:** Sam Sepassi  
**Course:** Udacity Data Science Nanodegree  
**Dataset:** IBM Watson Studio user-article interactions

---

## Problem Statement

The IBM Watson Studio platform hosts thousands of community articles on data science, machine learning, and AI. Users interact with articles, but there is no personalized recommendation system to help them discover relevant content they haven't seen yet.

This project builds a **multi-method recommendation engine** that analyzes user-article interaction data and recommends new articles to each user based on their reading history, similar users' behavior, article content similarity, and latent interest patterns.

---

## Key Features

- **Five recommendation methods** implemented and compared end-to-end
- **Custom user-item matrix** built from binary interaction data (5,148 users × 714 articles)
- **User-User Collaborative Filtering** with dot-product similarity, improved with popularity-based tie-breaking
- **Content-Based Filtering** using TF-IDF + KMeans clustering + cosine similarity on article text
- **Matrix Factorization (SVD)** for latent-factor article-article and user-article recommendations
- **Cold-start handling** for new users via rank-based fallback
- All rubric tests (`sol_1_test`, `sol_2_test`, `sol_4_test`, `sol_5_test`) called in notebook

---

## Project Structure

```
Recommendations_with_IBM/
├── Recommendations_with_IBM.ipynb   # Main notebook — fully executed with all outputs
├── project_tests.py                 # Udacity rubric test functions
├── data/
│   ├── user-item-interactions.csv   # 45,993 user-article interaction records
│   └── articles_community.csv       # 1,056 articles with full text content
├── top_5.p                          # Pickle: expected top-5 article names
├── top_10.p                         # Pickle: expected top-10 article names
├── top_20.p                         # Pickle: expected top-20 article names
└── README.md                        # This file
```

---

## Recommendation Methods

| Part | Method | Approach | Best For |
|------|--------|----------|----------|
| I | EDA | Explore users, articles, interaction distributions | — |
| II | Rank-Based | Most interacted articles globally | New users (cold start) |
| III | User-User CF | Dot-product similarity on binary interaction vectors | Returning users with history |
| IV | Content-Based | TF-IDF bigrams + KMeans (K=10) + cosine similarity | New articles; topic discovery |
| V | Matrix Factorization | SVD (K=15 latent features) on user-item matrix | Latent interest patterns |

---

## Key Results

| Metric | Value |
|---|---|
| Total interactions | 45,993 |
| Unique users | 5,148 |
| Unique articles (with interactions) | 714 |
| Total articles on platform | 1,051 |
| Median user interactions | 3 |
| Max interactions by one user | 364 |
| Most viewed article | 1429.0 (937 views) |
| Top-5 test | ✅ Passes |
| Top-10 test | ✅ Passes |
| Top-20 test | ✅ Passes |
| SVD latent features selected | 15 (elbow method) |
| KMeans optimal clusters | 10 (elbow method) |

---

## Libraries Used

| Library | Purpose |
|---|---|
| **pandas** | Data loading, manipulation, user-item matrix construction |
| **numpy** | Matrix operations, SVD reconstruction, error computation |
| **scipy** | Sparse SVD (`svds`) for efficient matrix factorization |
| **scikit-learn** | TF-IDF vectorization, KMeans clustering, cosine similarity |
| **matplotlib** | EDA charts, scree plot, elbow plots |
| **pickle** | Loading rubric test solution files |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/samsepassi1/School.git
cd School/ibm-recommendations

# 2. Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib jupyter

# 3. Launch the notebook
jupyter notebook Recommendations_with_IBM.ipynb
```

Run all cells with **Kernel → Restart & Run All**.

---

## Discussion

**Best method for this dataset:** User-User Collaborative Filtering (improved) — the interaction data is too sparse (~1.25% density) for SVD to capture strong latent signals.

**Production recommendation:** A **hybrid approach** combining:
1. User-User CF for returning users with sufficient history
2. Content-Based filtering for new articles not yet in the interaction matrix
3. Rank-Based fallback for new users (cold-start)

**How to evaluate in production:** A/B test CF vs. rank-based recommendations, measuring click-through rate, time-on-article, and return visit rate. Offline: hold out each user's most recent interactions and measure precision@K and recall@K.
