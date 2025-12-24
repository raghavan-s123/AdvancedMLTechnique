import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

salary_dummies = pd.get_dummies(df["salary"], prefix="salary")
df_salary = pd.concat([df, salary_dummies], axis = 1)

print("Creating dummy variables for salary:")
print(df_salary.head())
print()

department_dummies = pd.get_dummies(df_salary["Department"], prefix="dept")
df_final = pd.concat([df_salary, department_dummies], axis = 1)

print("Creating dummy variables for department:")
print(df_final.head())
print()

print("Final dataframe with dummy variables:")
print(df_final.head())
print()

train_df, test_df = train_test_split(df_final, test_size=0.3)

print(f"Size of training dataset: {train_df.shape}")
print(f"Size of test dataset: {test_df.shape}")

xtrain = train_df.drop(columns='left', axis=1)
ytrain = train_df['left']

xtest = test_df.drop(columns='left', axis=1)
ytest = test_df['left']

print("Shapes of input/output features after train-test split:")
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
