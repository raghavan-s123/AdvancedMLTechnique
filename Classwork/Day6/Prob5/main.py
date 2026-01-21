import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

filename = input().strip()

data = pd.read_csv(os.path.join(sys.path[0], filename))


X = data.drop("Fasting Blood Sugar Level", axis=1).values
y = data["Fasting Blood Sugar Level"].values

rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

y_true_ab = []
y_pred_ab = []

for train_idx, test_idx in rkf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = AdaBoostRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    y_true_ab.extend(y_test)
    y_pred_ab.extend(preds)

r2_ab = r2_score(y_true_ab, y_pred_ab)

print("=== AdaBoost Regressor Performance ===")
print(f"R-squared: {r2_ab:.3f}")
print()

y_true_st = []
y_pred_st = []

estimators = [
    ("lr", LinearRegression()),
    ("knn", KNeighborsRegressor()),
    ("dt", DecisionTreeRegressor(random_state=42))
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5
)

for train_idx, test_idx in rkf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    stack_model.fit(X_train, y_train)
    preds = stack_model.predict(X_test)

    y_true_st.extend(y_test)
    y_pred_st.extend(preds)

r2_st = r2_score(y_true_st, y_pred_st)

print("=== Stacking Regressor Performance ===")
print(f"R-squared: {r2_st:.3f}")
