import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("=== First 5 Rows of Data ===")
print(df.head())
print()

print(f"The number of samples in data is {len(df)}.")
print()

print("=== Data Types ===")
print(df.dtypes)
print()

print("=== Statistical Summary (Describe) ===")
print(df.describe())
print()

print("=== Missing Values Per Column ===")
print(df.isnull().sum())
print()

salary_encoder = LabelEncoder()
df['salary.enc'] = salary_encoder.fit_transform(df['salary'])

print("=== Salary Encoding Classes ===")
print(list(salary_encoder.classes_))
print()

dpt_encoder = LabelEncoder()
df['Department.enc'] = dpt_encoder.fit_transform(df['Department'])

print("=== Department Encoding Classes ===")
print(list(dpt_encoder.classes_))


df.drop(columns=['salary', 'Department'], inplace=True)
print("=== Dropping 'Department' and 'salary' columns ===")
print()

print("=== Updated DataFrame Info ===")
df.info()