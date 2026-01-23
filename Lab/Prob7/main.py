# Hierarchical Clustering on Iris Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# 1. Read the dataset
data = pd.read_csv("Resources/Iris(in).csv")

# 2. Extract only feature columns
features = data[['sepal_length', 'sepal_width',
                 'petal_length', 'petal_width']]

# 3. Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. Compute hierarchical clustering using Ward’s method
linked = linkage(scaled_features, method='ward')

# 5. Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Dendrogram for Iris Dataset (Ward's Method)")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# 6. Extract clusters (e.g., 3 clusters)
hc = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

cluster_labels = hc.fit_predict(scaled_features)

# 7. Print cluster labels
print("Cluster labels for each data point:")
print(cluster_labels)
