import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Resources/ML470_S7_Vehicle_Data_Practice.csv")

# -------------------------------------------------
# Drop missing values
# -------------------------------------------------
df = df.dropna()

# -------------------------------------------------
# Select numeric columns only
# -------------------------------------------------
X = df.select_dtypes(include=[np.number]).values

# -------------------------------------------------
# Standardize features
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =================================================
# 1️⃣ SILHOUETTE SCORE ANALYSIS (K = 2 to 9)
# =================================================
sil_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

plt.figure(figsize=(8, 5))
plt.bar(k_range, sil_scores)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Cluster Sizes")
plt.xticks(k_range)
plt.tight_layout()
plt.show()

# =================================================
# 2️⃣ HIERARCHICAL CLUSTERING – DENDROGRAM
# =================================================
linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(
    linked,
    orientation="right",
    distance_sort="descending",
    show_leaf_counts=True
)

# Horizontal cut line (visual threshold)
plt.axvline(x=15, color="blue", linestyle="--")

plt.title("Cars Dendrogram")
plt.xlabel("Distance")
plt.ylabel("Vehicle Index")
plt.tight_layout()
plt.show()
