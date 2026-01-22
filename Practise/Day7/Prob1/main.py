import pandas as pd
import os
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

file = input()

def outlier1():
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 - 1.5 * IQR
        
        outlier = df[(df[col] < low) | (df[col] > high)]
        print(f"{col}: {len(outlier)} outliers")
    
    

df = pd.read_csv(os.path.join(sys.path[0], file))

print("First 5 rows of the dataset:")
print(df.head())
print()

print("Dataset information:")
print(df.info())
print()

print("Missing values:")
print(df.isnull().sum())
print()

df.dropna(inplace=True)
print(f"Rows after removing missing values: {df.shape[0]}")
print()

df = df.select_dtypes(include=np.number)
print("Numeric columns:")
print(list(df.columns))
print()

print("Outlier summary using IQR method:")

outlier1()

for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    
    df[col] = df[col].clip(low, high)

print()
print("Outliers treated.")
print()

scaler = StandardScaler()

scaled = pd.DataFrame(scaler.fit_transform(df[['Width', 'Length']]), columns=['Width', 'Length'])
print("Scaled Width & Length:")
print(scaled.head())

print()
print("Preprocessing completed successfully.")


    
