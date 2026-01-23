# DBSCAN Clustering on Iris Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 1. Read the dataset
data = pd.read_csv("Resources/Iris(in).csv")

# 2. Extract feature columns
features = data[['sepal_length', 'sepal_width',
                 'petal_length', 'petal_width']]

# 3. Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(scaled_features)

# 5. Print cluster labels
print("Cluster labels for each sample:")
print(cluster_labels)

# 6. Number of clusters (excluding noise)
unique_clusters = set(cluster_labels)
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

# 7. Number of noise points
num_noise = list(cluster_labels).count(-1)

print("\nNumber of clusters found (excluding noise):", num_clusters)
print("Number of noise points:", num_noise)

# 8. Scatter plot (first two standardized features)
plt.figure(figsize=(8, 6))
plt.scatter(
    scaled_features[:, 0],
    scaled_features[:, 1],
    c=cluster_labels
)
plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.title("DBSCAN Clustering on Iris Dataset")
plt.show()
