import pandas as pd
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

print("Enter your data file (CSV or XLSX): ")
file = input()


df = pd.read_csv(os.path.join(sys.path[0], file))
print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())


print()

x = df.drop(columns='weight_condition_n')
y = df['weight_condition_n']

print("Input Data:")
print(x.head())
print()

print("Output Data:")
print(y.head())
print()

print("Scaling Input Data...")
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
print()

print("Silhouette Scores WITHOUT PCA:")
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=10)
    labels = kmeans.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    print(f"k={k}: Silhouette Score = {round(score, 3)}")

print("\nRunning KMeans WITHOUT PCA...")
kmeans = KMeans(n_clusters=2, random_state=10)
labels = kmeans.fit_predict(x_scaled)

print("\nCluster Evaluation:")
score_no_pca = silhouette_score(x_scaled, labels)
print(f"Silhouette Score: {round(score_no_pca, 3)}")

print("\nCluster Member Counts:")
print(pd.Series(labels).value_counts())

print("\nRunning PCA (n_components=2)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled)

print("\nSilhouette Scores WITH PCA:")
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}: Silhouette Score = {round(score, 3)}")

print("\nRunning KMeans WITH PCA...")
kmeans_pca = KMeans(n_clusters=2, random_state=10)
labels_pca = kmeans_pca.fit_predict(X_pca)

print("\nCluster Evaluation:")
score_pca = silhouette_score(X_pca, labels_pca)
print(f"Silhouette Score: {round(score_pca, 3)}")

print("\nCluster Member Counts:")
print(pd.Series(labels_pca).value_counts())

print("\n==================== SUMMARY ====================")
print("Data Loaded")
print("Data Scaled")
print("Optimal k checked using Silhouette Score")
print("K-Means applied WITHOUT PCA")
print("K-Means applied WITH PCA")
print("Evaluation completed using silhouette score")
print("==================================================")



