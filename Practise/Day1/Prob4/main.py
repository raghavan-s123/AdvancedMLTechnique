import pandas as pd
import os
import sys

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

missing = df.loc[df.isnull().any(axis=1)]

if missing.empty:
    print("Rows with missing values (if any):")
    print("No missing values found in the dataset.")
else:
    print("Rows with missing values (if any):")
    print(missing)

print()

new_df = df.select_dtypes(include=['number'])
corr = new_df.corr()
print("Correlation matrix of numeric columns:")
print(corr)

