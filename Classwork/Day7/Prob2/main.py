# ===============================
# Customer Segmentation Project
# K-Means & Agglomerative Clustering
# ===============================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("Resources/ML470_S7_MallCustomers_Data_Concept.csv")

# Select relevant features
df_features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# 2. Feature Scaling
# -------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# -------------------------------
# 3. Elbow Method
# -------------------------------
inertia = []

K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# -------------------------------
# 4. Silhouette Score Analysis
# -------------------------------
sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    sil_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.show()

# Best K based on silhouette score
best_k = K_range[np.argmax(sil_scores)]
print("Optimal number of clusters based on Silhouette Score:", best_k)

# -------------------------------
# 5. Final K-Means Model
# -------------------------------
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
df['KMeans_Cluster'] = kmeans_final.fit_predict(df_scaled)

# -------------------------------
# 6. Agglomerative Clustering
# -------------------------------
agg_model = AgglomerativeClustering(
    n_clusters=best_k,
    metric='euclidean',
    linkage='ward'
)

df['Agglomerative_Cluster'] = agg_model.fit_predict(df_scaled)

# -------------------------------
# 7. Pairplot Visualization
# -------------------------------
sns.pairplot(
    df,
    vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    hue='Agglomerative_Cluster',
    palette='tab10'
)
plt.show()

# -------------------------------
# 8. Cluster-wise Bar Charts
# -------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(
    x='Agglomerative_Cluster',
    data=df,
    palette='tab10'
)
plt.xlabel("Cluster")
plt.ylabel("Customer Count")
plt.title("Customer Distribution per Cluster")
plt.show()

# -------------------------------
# 9. Cluster Profile Summary
# -------------------------------
cluster_summary = df.groupby('Agglomerative_Cluster').mean()
print("\nCluster Characteristics (Mean Values):\n")
print(cluster_summary)

# -------------------------------
# 10. Model Evaluation
# -------------------------------
kmeans_silhouette = silhouette_score(df_scaled, df['KMeans_Cluster'])
agg_silhouette = silhouette_score(df_scaled, df['Agglomerative_Cluster'])

print("\nModel Performance:")
print("K-Means Silhouette Score:", round(kmeans_silhouette, 3))
print("Agglomerative Silhouette Score:", round(agg_silhouette, 3))

# -------------------------------
# End of Program
# -------------------------------
