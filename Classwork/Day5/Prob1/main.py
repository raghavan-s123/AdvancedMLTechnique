import pandas as pd
import os
import sys
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings(action="ignore")

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

x = df.drop(columns='target').values
y = df['target'].values

rskf = RepeatedStratifiedKFold(n_splits = 10, n_repeats=3, random_state=42)

clf = RandomForestClassifier(
        max_depth=5,
        n_estimators=100,
        n_jobs=-1,
        oob_score=True,
        random_state=42
    )

ytrueall = []
ypredall = []
for train_idx, test_idx in rskf.split(x, y):
    
    xtrain, xtest = x[train_idx], x[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    
    ytrueall.append(ytest)
    ypredall.append(ypred)

ytrueall = np.concatenate(ytrueall)
ypredall = np.concatenate(ypredall)

acc = accuracy_score(ytrueall, ypredall)


clf.fit(x, y)
oob = clf.oob_score_

print("=================================")
print(f"Accuracy: {acc:.3f}")
print(f"OOB Score: {oob:.3f}")
print("=================================")