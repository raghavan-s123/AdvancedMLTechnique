import pandas as pd
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score

def kfold(x, y, model):
    
    kf = KFold(n_splits = 5, shuffle=True, random_state=42)
    kf_scores = []

    for train_idx, test_idx in kf.split(x):
        
        xtrain, xtest = x[train_idx], x[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx]
        
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        kf_scores.append(accuracy_score(ytest, ypred))
        
    kf_scores = np.round(kf_scores, 3)
    print(f"k-Fold Accuracy Scores : {kf_scores}")
    print(f"Mean CV Accuracy: {np.mean(kf_scores):.3f}")

def holdout(x, y, model):
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
    
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    
    acc = accuracy_score(ytest, ypred)
    
    print(f"Hold-Out Method Accuracy: {acc:.3f}")
    
def leaveone(x, y, model):
    
    loo = LeaveOneOut()
    loo_preds=[]
    loo_true=[]
    
    for train_idx, test_idx in loo.split(x):
        
        xtrain, xtest = x[train_idx], x[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx]
        
        model.fit(xtrain, ytrain)
        pred = model.predict(xtest)
        loo_preds.append(pred[0])
        loo_true.append(ytest[0])
    
    loo_acc = accuracy_score(loo_true, loo_preds)
    print(f"LOOCV Accuracy: {loo_acc:.3f}")
    
def stratified(x, y, model):
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    skf_scores = []
    
    for train_idx, test_idx in skf.split(x, y.ravel()):
        
        xtrain, xtest = x[train_idx], x[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx]
        
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        skf_scores.append(accuracy_score(ytest, ypred))
    
    acc = np.mean(skf_scores)
    print(f"Accuracy: {acc:.3f}")

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print(df.head())
print()

x = df.drop(columns='target').values
y = df['target'].values

model = DecisionTreeClassifier(max_depth=4, random_state=42)

kfold(x, y, model)
holdout(x, y, model)
leaveone(x, y, model)
stratified(x, y, model)

