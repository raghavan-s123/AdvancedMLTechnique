import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("Dataset loaded successfully.")
print()

print("Preview of dataset:")
print(df.head())
print()

print("Dataset information:")
df.info()
print()

print("Missing values in each column:")

print(df.isnull().sum())
print()

x = df.drop(columns={"price_range"})
y = df['price_range']
print('Input and target variables separated.')
print()

corr = x.corr() > 0.7
print("Multicollinearity check:")
print(corr)
print()

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

print("Scaled input features (first 5 rows):")
print(x_scaled.head())