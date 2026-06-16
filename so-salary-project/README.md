# 💼 What Makes Developers Rich?
### Predicting Developer Salaries with the Stack Overflow 2024 Developer Survey

**Author:** Sam Sepassi  
**Blog Post:** [What Makes Developers Rich? (Medium / GitHub Pages)](https://samsepassi1.github.io/School)  

---

## Motivation

What drives a software developer's salary? Is it experience? Education? The country you live in? 
Or does going remote actually pay off? This project uses the Stack Overflow 2024 Developer Survey 
to build a machine learning model that predicts annual developer compensation and uncovers 
surprising, actionable insights about what really matters.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations, log transforms |
| `scikit-learn` | ML pipelines, preprocessing, modeling, evaluation |
| `matplotlib` | All data visualizations |
| `seaborn` | Plot theming and styling |

---

## Files in the Repository

| File | Description |
|---|---|
| `stackoverflow_salary_analysis.ipynb` | Main Jupyter notebook — full CRISP-DM analysis, EDA, modeling, and predictions |
| `stackoverflow_survey_2024_sample.csv` | 5,000-row Stack Overflow Developer Survey 2024 sample |
| `fig1_salary_dist.png` | Target variable distribution (raw + log-transformed) |
| `fig2_salary_country.png` | Median salary by country (top 10) |
| `fig3_salary_experience.png` | Salary vs years of professional experience |
| `fig4_salary_devtype.png` | Median salary by developer type |
| `fig5_salary_remote.png` | Salary by remote work arrangement (the surprise finding) |
| `fig6_ai_salary.png` | AI tool adoption vs salary |
| `fig7_model_comparison.png` | Ridge vs Random Forest vs Gradient Boosting |
| `fig8_actual_vs_predicted.png` | Best model: actual vs predicted salary scatter |
| `fig9_feature_importance.png` | Aggregated feature importance from Gradient Boosting |
| `fig10_creative_prediction.png` | Jordan's salary prediction across work arrangements |
| `README.md` | This file |

---

## Business Questions & Results Summary

| # | Question | Key Finding |
|---|---|---|
| 1 | What features drive salary? | **Country** explains 82% of salary variance. A US developer earns ~4x more than an Indian developer with identical skills and experience. |
| 2 | Creative / unusual insights? | **Remote workers earn MORE, not less.** Fully remote developers command a salary premium — not because remote work pays more, but because senior US engineers disproportionately work remotely (self-selection). |
| 3 | How accurate is the model? | **R² = 0.87** (Ridge Regression, best model). The model explains 87% of salary variance with a mean absolute error of ~$16,300. |
| 4 | Predictive scenario? | A senior US ML engineer (Jordan) going fully remote is predicted to earn **~$9,400 more** than the in-person equivalent — a quantified remote premium. |

---

## CRISP-DM Process

1. **Business Understanding** — defined 4 salary-related questions
2. **Data Understanding** — EDA: distributions, correlations, country/experience/role breakdowns  
3. **Data Preparation** — dropped missing targets (~12%), log-transformed salary, OHE for categoricals, StandardScaler for numerics  
4. **Modeling** — compared Ridge, Random Forest, and Gradient Boosting inside sklearn Pipelines  
5. **Evaluation** — R², MAE, RMSE on held-out test set; feature importance analysis  
6. **Deployment** — creative prediction scenario for a concrete, actionable takeaway  

---

## Acknowledgements

- Stack Overflow Developer Survey: https://survey.stackoverflow.co/
- scikit-learn documentation: https://scikit-learn.org/
- Udacity Data Scientist Nanodegree — Project 1 template
