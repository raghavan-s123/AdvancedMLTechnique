import pandas as pd
import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score
from ML_Modules import outlier

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("# Head of all columns")
print(df.head())
print()

print("# Data Types of all columns")
print(df.dtypes)
print()

subset = df[['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c', 'Outcome']]
print("# Working subset head")
print(subset.head())
print()

mean = subset.groupby('Outcome').mean()
print("# Mean values grouped by Outcome")
print(mean)
print()

print("# Null value check")
print(subset.isnull().sum())
print()

columns = ['BMI', 'Glucose', 'Age']
for col in columns:
    zero = (subset[col] == 0).sum()
    print(f"# Zero-value count for {col}")
    print(zero)
    print()

subset = subset[
    (subset['Glucose'] != 0) &
    (subset['BMI'] != 0) &
    (subset['Age'] != 0)
    ]
columns = ['Glucose', 'BMI']
for col in columns:
    # subset = subset[subset[col] != 0]
    zero = (subset[col] == 0).sum()
    print(f"# Zero-value count after removal: {col}")
    print(zero)
    print()
    
print("# Number of rows after zero-value removal")
print(len(subset))
print()

x = subset.drop(columns='Outcome')
y = subset['Outcome']

for col in x.columns:
    original_dtype = x[col].dtype
    
    Q1 = x[col].quantile(0.25)
    Q3 = x[col].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    x[col] = np.where(x[col] < lower, lower, x[col])
    x[col] = np.where(x[col] > upper, upper, x[col])
    

print("# Data after outlier treatment")

outlier(file)
print()

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

svc = SVC()
svc.fit(xtrain, ytrain)

ypred = svc.predict(xtest)

matrix = confusion_matrix(ytest, ypred)

print("# SVM Model Evaluation")
print("Confusion Matrix")
print(matrix)
print("===================")
print()

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

