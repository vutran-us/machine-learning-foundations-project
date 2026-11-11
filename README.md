# Adult Income Prediction — Machine Learning Foundations

Supervised **binary classification** predicting whether a person's annual income exceeds **$50,000**, using the UCI Adult (Census Income) dataset. Built as Capstone 4 of the Udacity AI Master's program.

## Highlights
- End-to-end scikit-learn `Pipeline` (imputation, one-hot encoding, scaling) fit on training data only, so no preprocessing leakage
- Logistic regression compared against a majority-class baseline
- Metrics chosen for a 76/24 class imbalance: precision, recall, F1, ROC-AUC, plus confusion matrix, ROC, and precision–recall curves
- Subgroup fairness audit (TPR/FPR by sex and race) with sex and race **excluded from model inputs**

## Results (held-out test set, 9,758 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority baseline | 0.761 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.855 | 0.737 | 0.615 | 0.671 | 0.909 |

Five-fold cross-validation F1 on the training set: **0.654 ± 0.011**.

**Fairness check:** true positive rate was 0.496 for women vs. 0.639 for men (14.2 pp gap), even though sex was not a model input; the `relationship` field (Husband/Wife) may act as a proxy. Race comparisons are limited by small group sizes.

## Repository contents

| File | Description |
|---|---|
| `modeling.ipynb` | Full workflow: load, inspect, clean, preprocess, train, evaluate, fairness audit, summary |
| `module_summary.pdf` | Analysis report with methodology, results, plain-language interpretation, limitations, and APA 7 references |
| `adult.csv` | UCI Adult data: `adult.data` + `adult.test` combined (48,842 rows × 15 columns, values unchanged) |
| `requirements.txt` | Pinned environment from `pip freeze` |

## Run it

```bash
git clone https://github.com/vutran-us/machine-learning-foundations-project.git
cd machine-learning-foundations-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab modeling.ipynb   # then Kernel → Restart Kernel and Run All Cells
```

## Limitations
- Data is from the 1994 Census; the $50K threshold is not inflation-adjusted, so results don't describe today's labor market.
- Logistic regression is additive and won't capture feature interactions without engineered terms.
- The 0.50 decision threshold treats missed high earners and false alarms as equally costly.
- Not intended for decisions about individuals.

## Dataset and license
Becker, B., & Kohavi, R. (1996). *Adult* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---
*Completed as a capstone project in the Udacity AI Master's program. This repository is not affiliated with or endorsed by Udacity.*
