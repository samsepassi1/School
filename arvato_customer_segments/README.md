# Arvato Customer Segments

**Author:** Sam Sepassi

Udacity unsupervised learning project: identify customer segments for a German mail-order company using demographic data from AZ Direct / Arvato Financial Solutions.

## Important data note

The project data is proprietary. Do **not** commit the CSV files to GitHub. If downloaded locally, delete the data after completing/submitting the project per the Arvato/Bertelsmann agreement.

Expected local-only files:

- `Udacity_AZDIAS_Subset.csv`
- `Udacity_CUSTOMERS_Subset.csv`
- `AZDIAS_Feature_Summary.csv`
- `Data_Dictionary.md`

## Submission files

Udacity expects a zip containing:

- `Identify_Customer_Segments.ipynb`
- `Identify_Customer_Segments.html`

## Local setup

```bash
cd arvato_customer_segments
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Identify_Customer_Segments.ipynb
```

## Rubric checklist

### Preprocessing

- [ ] Re-encode all known missing/unknown values as `NaN`.
- [ ] Assess missing values by column.
- [ ] Remove columns with high missingness and justify threshold.
- [ ] Identify patterns in missingness across columns.
- [ ] Process mixed-type features by engineering usable features or dropping originals.
- [ ] Process categorical features: keep/recode binary, one-hot/drop multi-level.
- [ ] Assess missing values by row.
- [ ] Split high-missing rows from low-missing rows and compare distributions.
- [ ] Create `clean_data()` function for both general and customer data.
- [ ] Keep only analysis-ready numeric columns.

### Feature transformation

- [ ] Impute remaining missing values.
- [ ] Apply feature scaling.
- [ ] Fit PCA and analyze explained variance.
- [ ] Justify selected number of components.
- [ ] Interpret at least three principal components using feature weights.

### Clustering

- [ ] Fit KMeans over multiple `k` values.
- [ ] Plot/report average point-centroid distances.
- [ ] Choose and justify final cluster count.
- [ ] Apply same cleaning, imputer, scaler, PCA, and KMeans pipeline to customers.
- [ ] Compare cluster proportions for customers vs general population.
- [ ] Identify overrepresented and underrepresented customer segments.
