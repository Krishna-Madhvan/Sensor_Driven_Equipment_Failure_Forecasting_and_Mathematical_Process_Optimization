# Sensor-Driven Equipment Failure Forecasting & Mathematical Process Optimization

Binary predictive-maintenance classification on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv) — 10,000 synthetic machine cycles with thermal, mechanical, and tool-wear sensor readings — comparing two approaches to catching failures before they happen:

1. **Standalone Random Forest** — class-weighted baseline classifier
2. **Multi-Model Voting Ensemble** — Random Forest + Gradient Boosting + Logistic Regression combined via soft voting

A from-scratch gradient-descent linear regression implementation (`MeraLR`) is also included as a transparent, mathematically-interpretable baseline for the underlying optimization mechanics that power the tree-based models.

---

## Notebook Structure

The full pipeline — data loading, EDA, feature engineering, model training, and evaluation — lives in a single self-contained notebook:

```
├── Sensor_Driven_Equipment_Failure_Forecasting_and_Mathematical_Process_Optimization.ipynb
└── README.md
```

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 logged machine cycles pulled directly from the UCI Machine Learning Repository at runtime.

**Target:** `Machine failure` (0 = normal operation, 1 = failure) — a severely imbalanced binary label (~3.4% positive class).

**Raw sensor features:**
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

### Feature Engineering

- `Temp_Diff` — process temperature minus air temperature (thermal stress proxy)
- `Power_Est` — rotational speed × torque (estimated mechanical power draw)
- `Rolling_Torque_Mean` — 5-cycle rolling mean of torque
- `Rolling_Speed_Std` — 5-cycle rolling standard deviation of rotational speed

Failure-mode indicator columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers (`UDI`, `Product ID`, `Type`) are dropped from the feature set to avoid leakage and keep the model purely sensor-driven.

- 80/20 stratified train/test split (seed = 42), preserving the failure/normal class ratio
- Features standardized with `StandardScaler`

---

## Model 1 — Standalone Random Forest

A class-weighted `RandomForestClassifier` trained directly on the engineered sensor features.

| Setting | Value |
|---|---|
| Estimators | 100 |
| Class weight | Balanced |
| Random state | 42 |

---

## Model 2 — Multi-Model Voting Ensemble

A soft-voting `VotingClassifier` combining three complementary learners so that no single model's blind spots dominate the failure call.

| Member | Role |
|---|---|
| Random Forest | Non-linear splits, robust to outliers |
| Gradient Boosting | Sequential error-correction on residual failures |
| Logistic Regression (balanced) | Linear, well-calibrated probability baseline |

Voting is **soft** (probability-averaged) rather than hard-majority, letting confident models outweigh uncertain ones on borderline cycles.

---

## Custom Optimizer — `MeraLR`

A gradient-descent linear regression class built from scratch in NumPy (no scikit-learn) to make the underlying "mathematical process optimization" explicit:

| Setting | Value |
|---|---|
| Learning rate | 0.01 |
| Epochs | 1000 |
| Update rule | Batch gradient descent on MSE |

---

## Evaluation

**Metrics:** precision, recall, F1-score, confusion matrix, ROC curve / AUC, and false-positive rate — chosen over raw accuracy because the failure class is rare and a missed failure is far costlier than a false alarm.

| Model | Recall (failures caught) | Precision | False Alarm Rate |
|---|---|---|---|
| Standalone Random Forest | 64.71% | 91.67% | 0.21% |
| Voting Ensemble | **79.41%** | 88.52% | 0.36% |

The ensemble trades a small amount of precision and a marginally higher false-alarm rate for a **~15-point jump in recall** — catching substantially more real failures, which is the priority in a plant-safety setting.

The notebook also plots top engineered-feature importances from the Random Forest to surface which sensor signals drive failure predictions.

---

## Setup

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Dataset

The notebook pulls the dataset directly from the UCI repository at runtime — no manual download needed:

```python
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")
```

---

## References

- Stephan Matzka, [Explainable Artificial Intelligence for Predictive Maintenance Applications](https://ieeexplore.ieee.org/document/9253083), AI4I 2020
- [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- [scikit-learn: VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
