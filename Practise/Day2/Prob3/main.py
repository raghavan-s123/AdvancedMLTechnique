import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

salary_encoder = LabelEncoder()
dept_encoder = LabelEncoder()

df['salary.enc'] = salary_encoder.fit_transform(df['salary'])
df['Department.enc'] = dept_encoder.fit_transform(df['Department'])

print("=== Label Encoding Categorical Columns ===")
print(f"Encoded salary classes: {list(salary_encoder.classes_)}")
print(f"Encoded Department classes: {list(dept_encoder.classes_)}")
print()


df.drop(columns=['Department','salary'], inplace=True)
x = df.drop(columns='left')
y = df['left']

print("=== Separating Features and Label ===")
print(f"Input Features Shape: {x.shape}")
print(f"Label Shape: {y.shape}")
print()

matrix = x.corr()
corr = matrix >= 0.75
print("=== Correlation Boolean Matrix (correlation >= 0.75) ===")
print(corr)
print()

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

print("=== Scaled Feature Sample (First 5 Rows) ===")
print(x_scaled.head())
print()


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state=42)
print("=== Splitting Data into Train (80%) and Test (20%) ===")
print(f"Training Features Shape: {xtrain.shape}")
print(f"Training Labels Shape: {ytrain.shape}")
print(f"Testing Features Shape: {xtest.shape}")
print(f"Testing Labels Shape: {ytest.shape}")


