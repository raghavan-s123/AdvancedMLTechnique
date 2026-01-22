import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN



df = pd.read_csv("Resources/ML470_S8_Wholesale Customers data_Practice.csv")

# Select numerical columns and round values
num_df = df.select_dtypes(include=np.number).round(2)

# ----------------------------------
# IQR-based Outlier Treatment
# ----------------------------------

for col in num_df.columns:
    Q1 = num_df[col].quantile(0.25)
    Q3 = num_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    num_df[col] = num_df[col].clip(lower, upper)

# ----------------------------------
# Remove Highly Correlated Feature
# ----------------------------------

# As specified in the problem statement
if "Detergents_Paper" in num_df.columns:
    num_df = num_df.drop(columns=["Detergents_Paper"])

# ----------------------------------
# Standardize Numerical Features
# ----------------------------------

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

# ----------------------------------
# 2. Apply DBSCAN Clustering
# ----------------------------------

dbscan = DBSCAN(eps=2, min_samples=5)
labels = dbscan.fit_predict(scaled_data)

# Add cluster labels to original dataframe
df["cluster"] = labels

# Display final cluster count summary
print(df["cluster"].value_counts())

# ----------------------------------
# 3. Visualization — DBSCAN Cluster Plot
# ----------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(
    df["Fresh"],
    df["Grocery"],
    c=df["cluster"],
    alpha=0.8
)
plt.title("DBSCAN Cluster Plot (Fresh vs Grocery)")
plt.xlabel("Fresh")
plt.ylabel("Grocery")
plt.xlim(0, 35000)
plt.ylim(0, 20000)

plt.tight_layout()
plt.show()
