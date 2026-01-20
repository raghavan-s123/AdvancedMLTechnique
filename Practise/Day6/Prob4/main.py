import pandas as pd
import os
import sys
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score


warnings.simplefilter("ignore")

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

x = df.drop(columns={'price_range'}).values
y = df['price_range'].values

scaler = StandardScaler()
x_scaled = (scaler.fit_transform(x))

base_models = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("knn", KNeighborsClassifier()),
        ("dt", DecisionTreeClassifier(random_state=42))
    ]

learner = LogisticRegression(max_iter=1000, random_state=42)

stacking = StackingClassifier(
        estimators = base_models,
        final_estimator=learner,
        cv=5
    )
    
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

ytrue = []
ypred = []


for train_idx, test_idx in rskf.split(x_scaled, y):
    
    xtrain, xtest = x_scaled[train_idx], x_scaled[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    stacking.fit(xtrain, ytrain)
    preds = stacking.predict(xtest)
    
    ytrue.extend(ytest)
    ypred.extend(preds)
    
    
ytrue = np.array(ytrue)
ypred = np.array(ypred)

acc = accuracy_score(ytrue, ypred)
print(f"Accuracy: {acc:.3f}")
print()

con = confusion_matrix(ytrue, ypred)
print("Confusion Matrix")
print(con)
print("===================")

print("Classification Report:")
print(classification_report(ytrue, ypred, digits=2))

print("===================")

rec = recall_score(ytrue, ypred, average="weighted")
f1 = f1_score(ytrue, ypred, average="weighted")
prec = precision_score(ytrue, ypred, average="weighted")

print(f"accuracy: {acc:.3f}")
print(f"recall: {rec:.3f}")
print(f"f1-score: {f1:.3f}")
print(f"precision: {prec:.3f}")

