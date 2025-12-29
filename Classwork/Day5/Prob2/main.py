import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score


df = pd.read_excel("Resources/ML470_S5_Diabetes_Cleaned_Data_Concept.xlsx")

target_col = "target"

X = df.drop(columns=[target_col])
y = df[target_col]

X = X.select_dtypes(include=["int64", "float64"])


cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

y_true_all = []
y_prob_all = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    y_true_all.extend(y_test)
    y_prob_all.extend(y_prob)

y_true_all = np.array(y_true_all)
y_prob_all = np.array(y_prob_all)


fpr, tpr, thresholds = roc_curve(y_true_all, y_prob_all)
auc_score = roc_auc_score(y_true_all, y_prob_all)


plt.figure(figsize=(8, 7))

plt.plot(
    fpr,
    tpr,
    color="orange",
    linewidth=2,
    marker="o",
    markersize=4,
    label=f"Random Forest (AUC = {auc_score:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="blue",
    label="No Skill"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest Diabetes Prediction")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
