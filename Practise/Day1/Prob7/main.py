import os
import sys
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

salary_dummies = pd.get_dummies(df['salary'], prefix='salary')
dept_dummies = pd.get_dummies(df['Department'], prefix='dept')

df_encoded = pd.concat([df, salary_dummies, dept_dummies], axis=1)

df_encoded = df_encoded.drop(columns=['salary', 'Department'])

x = df_encoded.drop(columns='left', axis=1)
y = df['left']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
gnb = GaussianNB()

gnb.fit(xtrain, ytrain)
print(gnb)
print()

ypred = gnb.predict(xtest)
print(ypred)
