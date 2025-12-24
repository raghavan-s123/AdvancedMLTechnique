import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score


def evaluator():
    file = input()
    df = pd.read_csv(os.path.join(sys.path[0], file))
    
    salary_dummies = pd.get_dummies(df['salary'], prefix='salary')
    dept_dummies = pd.get_dummies(df['Department'], prefix='dept')
    
    df_encode = pd.concat([df, salary_dummies, dept_dummies], axis=1)
    
    df_encode = df_encode.drop(columns=['salary', 'Department'])
    
    x = df_encode.drop(columns='left', axis=1)
    y = df_encode['left']
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
    gnb = GaussianNB()
    
    gnb.fit(xtrain, ytrain)
    
    ypred = gnb.predict(xtest)
    
    matrix = confusion_matrix(ytest, ypred)
    print("Confusion Matrix")
    print(matrix)
    print("===================")
    
    report = classification_report(ytest, ypred)
    print("Classification Report:")
    print(report)
    print("===================")
    
    acc = accuracy_score(ytest, ypred)
    rec = recall_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    prec = precision_score(ytest, ypred)
    
    print(f"accuracy: {acc:.3f}")
    print(f"recall: {rec:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {prec:.3f}")
    
    print("code is available inside the 'ML_Modules.py' file")
        
        