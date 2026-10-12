# Mobile-money fraud detection · five classifiers on 6.36M transactions

Which model actually catches fraud when fraud is 0.13% of transactions? This notebook compares five
classifiers on the PaySim synthetic mobile-money dataset (6,362,620 transactions) and judges them on
recall and F1 for the fraud class, not on accuracy, which every model maxes out by predicting "not fraud".

## Results on the held-out test set (1,908,786 transactions, 2,468 frauds)

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | Accuracy |
| --- | --- | --- | --- | --- |
| **XGBoost** | **0.96** | 0.85 | **0.90** | 0.9998 |
| Decision tree | 0.89 | **0.88** | 0.88 | 0.9997 |
| Random forest | 0.98 | 0.78 | 0.87 | 0.9997 |
| Logistic regression | 0.92 | 0.42 | 0.58 | 0.9992 |
| Naive Bayes | 0.07 | 0.10 | 0.08 | 0.9970 |

**Takeaways**

- Accuracy is useless here: even the worst model scores 99.7%. Fraud-class recall and F1 separate the models.
- XGBoost gives the best balance, catching 85% of frauds with 96% of its alerts being real fraud.
- The decision tree catches slightly more fraud (88%) but raises more false alarms.
- Logistic regression misses more than half of all fraud; Naive Bayes is not usable on this data.

> The summary table inside the notebook repeats the decision-tree scores in the Naive Bayes row; the numbers
> above are taken from each model's own classification report.

## Approach

1. Exploratory analysis of transaction types, amounts and balance changes.
2. Encode transaction type, drop identifier columns, train/test split (70/30).
3. Fit logistic regression, random forest, decision tree, Gaussian Naive Bayes and XGBoost.
4. Compare precision, recall and F1 on the fraud class plus confusion matrices.

## Run it

1. Download the [PaySim dataset from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and save it as `data.csv` next to the notebook.
2. Install and launch:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

## Next steps

- Handle the class imbalance explicitly (class weights, SMOTE) and tune the decision threshold for a target recall
- Time-based split instead of a random split, so the model is tested on future transactions
- Precision-recall curves and cost-weighted evaluation (a missed fraud costs more than a false alert)

**Author:** Yuvraj Chawla · [yuvrajchawla.si](https://yuvrajchawla.si) · MIT License
