# Data Dictionary — CDC Diabetes Health Indicators (Balanced Subset)

This dataset is a balanced subset of the CDC Behavioral Risk Factor Surveillance
System (BRFSS) 2015 survey. Each row is one respondent.

The original survey has ~14% diabetes prevalence; this file is downsampled to a
50/50 class balance with roughly 50,000 rows so students can focus on the
neural-network fundamentals before tackling class imbalance.

## Target

| Column           | Type   | Description                                   |
|------------------|--------|-----------------------------------------------|
| `Diabetes_binary` | int (0/1) | 0 = no diabetes, 1 = pre-diabetes or diabetes |

## Features (21)

| Column                  | Type             | Range / values                                                                                            | Description                                                            |
|-------------------------|------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `HighBP`                | int (0/1)        | 0 = no, 1 = yes                                                                                           | Adult told they have high blood pressure                               |
| `HighChol`              | int (0/1)        | 0 = no, 1 = yes                                                                                           | Adult told they have high cholesterol                                  |
| `CholCheck`             | int (0/1)        | 0 = no check in 5 years, 1 = yes                                                                          | Cholesterol check in the last five years                               |
| `BMI`                   | float            | 12 – 98                                                                                                   | Body mass index                                                        |
| `Smoker`                | int (0/1)        | 0 = no, 1 = yes                                                                                           | Smoked at least 100 cigarettes in entire life                          |
| `Stroke`                | int (0/1)        | 0 = no, 1 = yes                                                                                           | Ever told had a stroke                                                 |
| `HeartDiseaseorAttack`  | int (0/1)        | 0 = no, 1 = yes                                                                                           | Coronary heart disease or myocardial infarction                        |
| `PhysActivity`          | int (0/1)        | 0 = no, 1 = yes                                                                                           | Physical activity in past 30 days (excluding job)                      |
| `Fruits`                | int (0/1)        | 0 = no, 1 = yes                                                                                           | Consumes fruit one or more times per day                               |
| `Veggies`               | int (0/1)        | 0 = no, 1 = yes                                                                                           | Consumes vegetables one or more times per day                          |
| `HvyAlcoholConsump`     | int (0/1)        | 0 = no, 1 = yes                                                                                           | Heavy drinker (adult men >14 drinks/week, adult women >7 drinks/week)  |
| `AnyHealthcare`         | int (0/1)        | 0 = no, 1 = yes                                                                                           | Has any kind of health-care coverage                                   |
| `NoDocbcCost`           | int (0/1)        | 0 = no, 1 = yes                                                                                           | Needed to see a doctor but could not because of cost in past 12 months |
| `GenHlth`               | int (1 – 5)      | 1 = excellent, 5 = poor                                                                                   | Self-rated general health                                              |
| `MentHlth`              | int (0 – 30)     | days                                                                                                       | Days of poor mental health in past 30 days                             |
| `PhysHlth`              | int (0 – 30)     | days                                                                                                       | Days of poor physical health in past 30 days                           |
| `DiffWalk`              | int (0/1)        | 0 = no, 1 = yes                                                                                           | Serious difficulty walking or climbing stairs                          |
| `Sex`                   | int (0/1)        | 0 = female, 1 = male                                                                                      | Biological sex                                                         |
| `Age`                   | int (1 – 13)     | 1 = 18–24, 2 = 25–29, … , 13 = 80+                                                                        | 13-level age category (BRFSS `_AGEG5YR`)                               |
| `Education`             | int (1 – 6)      | 1 = never attended, 2 = elementary, 3 = some HS, 4 = HS grad, 5 = some college, 6 = college grad          | Highest grade or year of school completed                              |
| `Income`                | int (1 – 8)      | 1 = <\$10k, 2 = <\$15k, …, 7 = <\$75k, 8 = ≥\$75k                                                          | Annual household income bracket                                        |

## Clinical context

Diabetes prevalence rises sharply with age, BMI, and the cluster of metabolic
risk factors (high blood pressure + high cholesterol + low physical activity).
A pre-screening model is expected to flag high-risk individuals so they can be
prioritised for diagnostic testing — false negatives (missed diabetics) are
generally considered more costly than false positives (extra tests).
