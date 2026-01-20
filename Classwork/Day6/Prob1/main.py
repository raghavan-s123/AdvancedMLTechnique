import pandas as pd
import numpy as np
import os
import sys
import warnings

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

warnings.filterwarnings("ignore")

filename = input().strip()
path = os.path.join(sys.path[0], filename)

if not os.path.exists(path):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

df = pd.read_csv(path)


print(df.head())
print()


X = df.drop(columns="Diabetic").values
y = df["Diabetic"].values

rkf = RepeatedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=42
)

acc_scores = []
rec_scores = []
f1_scores = []
prec_scores = []

for train_idx, test_idx in rkf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = AdaBoostClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc_scores.append(accuracy_score(y_test, preds))
    rec_scores.append(recall_score(y_test, preds, average="weighted"))
    f1_scores.append(f1_score(y_test, preds, average="weighted"))
    prec_scores.append(precision_score(y_test, preds, average="weighted"))


final_model = AdaBoostClassifier(random_state=42)
final_model.fit(X, y)
final_preds = final_model.predict(X)


print("Confusion Matrix")
print(confusion_matrix(y, final_preds))
print("===================\n")


print("Classification Report:")
print(classification_report(y, final_preds, digits=2))
print("===================")


print(f"accuracy:  {np.mean(acc_scores):.3f}")
print(f"recall: {np.mean(rec_scores):.3f}")
print(f"f1-score: {np.mean(f1_scores):.3f}")
print(f"precision: {np.mean(prec_scores):.3f}")

