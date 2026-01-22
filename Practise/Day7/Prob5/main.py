import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")


df = pd.read_csv("Resources/ML470_S7_Vehicle_Data_Practice.csv")
df = df.dropna()

# --------------------------------------------------
# Select numeric features for clustering
# --------------------------------------------------
features = ["Horsepower", "Curb_weight", "Width", "Length"]
X = df[features].values

# --------------------------------------------------
# Scale features
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# Agglomerative Clustering
# --------------------------------------------------
model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

clusters = model.fit_predict(X_scaled)
df["Cluster"] = clusters

# --------------------------------------------------
# 1️⃣ Pairplot (corner=True)
# --------------------------------------------------
# --------------------------------------------------
# 1️⃣ Pairplot – ALL numeric columns
# --------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove("Cluster")

sns.pairplot(
    df,
    vars=numeric_cols,
    hue="Cluster",
    corner=True,
    palette="viridis"
)
plt.show()


# --------------------------------------------------
# 2️⃣ Cluster-wise statistics
# --------------------------------------------------
for c in sorted(df["Cluster"].unique()):
    print(f"\nCluster {c}")
    print("-" * 20)
    print("Average Horsepower:", df[df["Cluster"] == c]["Horsepower"].mean())
    print("Average Curb Weight:", df[df["Cluster"] == c]["Curb_weight"].mean())

    if "Manufacturer" in df.columns:
        print("Unique Manufacturers:")
        print(df[df["Cluster"] == c]["Manufacturer"].unique())

# --------------------------------------------------
# 3️⃣ Manufacturer distribution (Horizontal Bar Charts)
# --------------------------------------------------
if "Manufacturer" in df.columns:
    clusters = sorted(df["Cluster"].unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(12, 5))

    if len(clusters) == 1:
        axes = [axes]

    for i, c in enumerate(clusters):
        counts = df[df["Cluster"] == c]["Manufacturer"].value_counts()
        axes[i].barh(counts.index, counts.values)
        axes[i].set_title(f"Cluster {c}")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 4️⃣ Cluster Evaluation Metrics
# --------------------------------------------------
sil = silhouette_score(X_scaled, clusters)
db = davies_bouldin_score(X_scaled, clusters)
ch = calinski_harabasz_score(X_scaled, clusters)

print("\nCluster Evaluation Metrics")
print("--------------------------")
print(f"Silhouette Score: {sil:.3f}")
print(f"Davies-Bouldin Index: {db:.3f}")
print(f"Calinski-Harabasz Score: {ch:.3f}")
