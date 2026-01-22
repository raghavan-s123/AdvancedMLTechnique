import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("Dataset Preview:")
print(df.head())
print()

print("Dataset Info:")
print(df.info())
print()

print("Dataset Description:")
print(df.describe())
print()

print("Missing Values:")
print(df.isnull().sum())
print()

df = df.astype('float')
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[col] = df[col].clip(lower, upper)
    
df = df.round(2)

print("Data After Outlier Treatment:")
print(df.head())
print()


corr = df.corr().abs()
matrix = corr >= 0.7

print("Multicollinearity Matrix:")
print(matrix)
print()

df.drop(columns=['Detergents_Paper'], inplace=True)

print("Columns after removal:")
print(list(df.columns))
print()

scaler = StandardScaler()

scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Scaled Data Preview:")
print(scaled.head())


