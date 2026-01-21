import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# -------------------------------------------------
df = pd.read_csv("Resources/ML470_S7_Vehicle_Data_Practice.csv")

# -------------------------------------------------
# Drop missing values
# -------------------------------------------------
df = df.dropna()

# -------------------------------------------------
# Extract Width and Length
# -------------------------------------------------
X = df[["Width", "Length"]].values

# -------------------------------------------------
# Standardize features
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Agglomerative Clustering
# -------------------------------------------------
agg = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

labels = agg.fit_predict(X_scaled)

# -------------------------------------------------
# Scatter Plot (PORTAL SAFE)
# -------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    X[labels == 0, 0],
    X[labels == 0, 1],
    c="purple",
    label="0",
    s=50
)

plt.scatter(
    X[labels == 1, 0],
    X[labels == 1, 1],
    c="gold",
    label="1",
    s=50
)

plt.xlabel("Car Width")
plt.ylabel("Car Length")
plt.title("Car Sales Clustering by Width and Length")
plt.legend(title="Clusters")
plt.tight_layout()
plt.show()
