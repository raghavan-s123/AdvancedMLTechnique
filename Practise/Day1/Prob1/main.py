import pandas as pd
import os
import sys

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("First 5 rows of the dataset:")
print(df.head())
print()

print("Number of samples in the data:")
print(len(df))
print()

print("Data types of each column:")
print(df.dtypes)
print()

print("Feature columns used for classification:")

new_df = df.copy()
new_df.drop(columns={'Department', 'salary', 'left'}, inplace=True)
print(list(new_df.columns))
print()

sum_df = df.copy()
sum_df.drop(columns={'Department', 'salary'}, inplace=True)
print("Statistical summary of numeric columns:")
print(sum_df.describe())
print()

cat_data = df[['Department', 'salary', 'left']]
print("Sample categorical data (Department, salary, left):")
print(cat_data.head())