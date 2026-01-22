import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN



df = pd.read_csv("Resources/ML470_S8_Wholesale Customers data_Practice.csv")

# Select numerical columns and round
num_df = df.select_dtypes(include=np.number).round(2)

# ----------------------------------
# 2. IQR-based Outlier Treatment
# ----------------------------------

for col in num_df.columns:
    Q1 = num_df[col].quantile(0.25)
    Q3 = num_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    num_df[col] = num_df[col].clip(lower, upper)

# ----------------------------------
# 3. Remove Highly Correlated Features
# ----------------------------------

corr = num_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.85)]
num_df = num_df.drop(columns=drop_cols)

# ----------------------------------
# 4. Standardize Data
# ----------------------------------

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

# ----------------------------------
# 5. K-distance Graph (k = 5)
# ----------------------------------

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)

k_distances = np.sort(distances[:, 4])

plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.axhline(y=0.6, linestyle='--')
plt.title("K-distance Graph for eps Selection")
plt.xlabel("Data Points sorted by distance")
plt.ylabel("5th Nearest Neighbor Distance")
plt.tight_layout()
plt.show()

# ----------------------------------
# 6. DBSCAN Clustering
# ----------------------------------

dbscan = DBSCAN(eps=2, min_samples=5)
labels = dbscan.fit_predict(scaled_data)

df["Cluster"] = labels

# ----------------------------------
# 7. DBSCAN Scatter Plot (Fresh vs Grocery)
# ----------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(
    df["Fresh"],
    df["Grocery"],
    c=labels,
    alpha=0.8
)
plt.title("DBSCAN Clustering Scatter Plot")
plt.xlabel("Fresh")
plt.ylabel("Grocery")
plt.tight_layout()
plt.show()
