import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt


df = pd.read_csv("Resources/ML470_S7_Vehicle_Data_Practice.csv")

# -----------------------------
# Display first rows
# -----------------------------
print(df.head())
print()

# -----------------------------
# Drop missing values
# -----------------------------
df = df.dropna()

# -----------------------------
# Select numeric columns
# -----------------------------
num_df = df.select_dtypes(include=[np.number])

# -----------------------------
# IQR-based Winsorization
# -----------------------------
for col in num_df.columns:
    Q1 = num_df[col].quantile(0.25)
    Q3 = num_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    num_df[col] = np.where(num_df[col] < lower, lower, num_df[col])
    num_df[col] = np.where(num_df[col] > upper, upper, num_df[col])

# -----------------------------
# BOXPLOT (matches portal image)
# -----------------------------
plt.figure(figsize=(14, 6))
plt.boxplot(
    num_df.values,
    labels=num_df.columns,
    patch_artist=True,
    showfliers=True
)
plt.xticks(rotation=45, ha="right")
plt.title("Box Plot of Vehicle Dataset Features")
plt.tight_layout()
plt.show()
