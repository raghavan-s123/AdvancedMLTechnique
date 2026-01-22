# DBSCAN Customer Segmentation with Visual Diagnostics
# ---------------------------------------------------
# Visualizations included:
# 1. K-distance graph for eps selection
# 2. Initial DBSCAN clustering (moderate parameters)
# 3. Final DBSCAN clustering (optimized parameters)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# ---------------------------------------------------
# 1. LOAD AND PREPARE DATASET
# ---------------------------------------------------

# Load dataset (update filename if needed)
df = pd.read_excel("Resources/ML470_S8_Customer_Data_Concept.xlsx")

# Select only numerical columns
num_df = df.select_dtypes(include=np.number)

# Standardize numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)


# ---------------------------------------------------
# 2. VISUALIZATION 1 — K-DISTANCE GRAPH (eps selection)
# ---------------------------------------------------

# Compute 2nd nearest neighbor distances
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)

# Sort distances
k_distances = np.sort(distances[:, 1])

# Plot K-distance graph
plt.figure()
plt.plot(k_distances)
plt.axhline(y=1.2, linestyle="--")
plt.xlabel("Sorted Data Points")
plt.ylabel("2nd Nearest Neighbor Distance")
plt.title("K-Distance Graph for Epsilon Selection")
plt.show()


# ---------------------------------------------------
# 3. VISUALIZATION 2 — INITIAL DBSCAN CLUSTERING
#    eps = 1.2, min_samples = 5
# ---------------------------------------------------

dbscan_initial = DBSCAN(eps=1.2, min_samples=5)
initial_labels = dbscan_initial.fit_predict(scaled_data)

df["Initial_Cluster"] = initial_labels

plt.figure()
plt.scatter(
    df["credit_limit"],
    df["installments_purchases"],
    c=initial_labels
)
plt.xlabel("Credit Limit")
plt.ylabel("Installments Purchases")
plt.title("Initial DBSCAN Clustering (eps=1.2, min_samples=5)")
plt.show()


# ---------------------------------------------------
# 4. VISUALIZATION 3 — FINAL DBSCAN CLUSTERING
#    eps = 2.0, min_samples = 3
# ---------------------------------------------------

dbscan_final = DBSCAN(eps=2.0, min_samples=3)
final_labels = dbscan_final.fit_predict(scaled_data)

df["Final_Cluster"] = final_labels

plt.figure()
plt.scatter(
    df["credit_limit"],
    df["installments_purchases"],
    c=final_labels
)
plt.xlabel("Credit Limit")
plt.ylabel("Installments Purchases")
plt.title("Final DBSCAN Clustering (eps=2, min_samples=3)")
plt.show()
