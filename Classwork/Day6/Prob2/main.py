import pandas as pd
import os
import sys
import warnings
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score, precision_score

warnings.simplefilter("ignore")

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

x = df.drop(columns='Diabetic').values
y = df['Diabetic'].values


basemodels = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier())
    ]
meta = LogisticRegression(max_iter=1000, random_state=42)

stacking = StackingClassifier(
        basemodels,
        meta,
        cv=5
    )

rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=3, 
        random_state=42
    )
    
ytrue = []
ypred = []

for trainidx, testidx in rkf.split(x):
    
    xtrain, xtest = x[trainidx], x[testidx]
    ytrain, ytest = y[trainidx], y[testidx]
    
    stacking.fit(xtrain,ytrain)
    
    preds = stacking.predict(xtest)
    
    ytrue.extend(ytest)
    ypred.extend(preds)
    
ytrue = np.array(ytrue)
ypred = np.array(ypred)

acc = accuracy_score(ytrue, ypred)
rec = recall_score(ytrue, ypred, average="weighted")
f1 = f1_score(ytrue, ypred, average="weighted")
prec = precision_score(ytrue, ypred, average="weighted")

con = confusion_matrix(ytrue, ypred)

print(f"Accuracy: {acc:.3f}")
print()

print("Confusion Matrix")
print(con)
print("===================")
print("Classification Report:")
print(classification_report(ytrue, ypred, digits=2))
print("===================")

print(f"accuracy: {acc:.3f}")
print(f"recall: {rec:.3f}")
print(f"f1-score:{f1:.3f}")
print(f"precision: {prec:.3f}")
