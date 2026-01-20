import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

filename = input()

data = pd.read_csv(os.path.join(sys.path[0], filename))


X = data.drop("price_range.enc", axis=1).values
y = data["price_range.enc"].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

base_learners = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier()),
    ("dt", DecisionTreeClassifier(random_state=42))
]

meta_learner = LogisticRegression(max_iter=1000)


stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)


stack_model.fit(X_train, y_train)


y_pred = stack_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print(f"Accuracy: {acc:.3f}")
print()

print("Confusion Matrix")
print(cm)
print("===================")
print()


print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=2
    )
)
print("===================")


precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"accuracy: {acc:.3f}")
print(f"recall: {recall:.3f}")
print(f"f1-score: {f1:.3f}")
print(f"precision: {precision:.3f}")
