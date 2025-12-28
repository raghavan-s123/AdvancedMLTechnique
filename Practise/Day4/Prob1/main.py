import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print(f"The number of samples in data is {len(df)}.")
print()

print("Data Types:")
print(df.dtypes)
print()

print("Numeric Summary:")
print(df.describe())
print()

df.drop(columns=['salary', 'Department'], inplace=True)
print("Data After Dropping Irrelevant Columns:")
print(df.info())
print()

x = df.drop(columns='average_monthly_hours')
y = df['average_monthly_hours']

print("Input Features:")
print(x.head())
print()

print("Target Variable:")
print(y.head())
print()

scaler = StandardScaler()

x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

print("Scaled Feature Data:")
print(x_scaled.head())

