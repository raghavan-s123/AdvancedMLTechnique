import pandas as pd
import os
import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from ML_Modules import evaluate_classifier

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print(df.head())
print()
print(df.dtypes)

x = np.array(df[['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']])
y = df['Outcome']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.33, random_state=42)

gnb = GaussianNB()
gnb.fit(xtrain, ytrain)

print()
print("Model trained.");
print()

ypred = gnb.predict(xtest)
print(f"Predicted Values: array({list(np.array(ypred))})")
print()

evaluate_classifier(ytest, ypred)



