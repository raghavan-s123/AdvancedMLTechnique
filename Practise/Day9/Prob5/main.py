import os
import sys
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.simplefilter(action='ignore')


def main():
   
    filename = input("Enter your dataset filename (CSV): ").strip()
    file_path = os.path.join(sys.path[0], filename)

   
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    
    print("First 5 rows of the dataset:")
    print(df.head())
    print()

    
    print("Number of samples and features:")
    print(df.shape)
    print()

    
    print("Data types of each column:")
    print(df.dtypes)
    print()

    
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        print("Error: No numeric columns found for clustering.")
        sys.exit(1)

    print("Numeric columns used for clustering:")
    print(df_numeric.columns.tolist())
    print()

    X = df_numeric.values

    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)
    print("Original shape:", X.shape)
    print("Reduced shape after PCA:", reduced_data.shape)
    print()

   
    print("Silhouette Scores for different cluster sizes:")
    for k in range(2, 10):
        temp_model = KMeans(n_clusters=k, random_state=10)
        cluster_labels = temp_model.fit_predict(reduced_data)
        score = round(silhouette_score(reduced_data, cluster_labels), 3)
        print(f"For n_clusters = {k}, Average Silhouette Score = {score}")
    print()

    # ============================
    # Step 8: Build KMeans Model with reduced dimensions
    # ============================
    final_k = 3
    kmeans_dr = KMeans(n_clusters=final_k, random_state=10)
    cluster_labels = kmeans_dr.fit_predict(reduced_data)
    print(f"Cluster labels assigned (K={final_k}):")
    print(cluster_labels)
    print()

   
    try:
        import ML_Modules as mm
        mm.evaluate_clusterer(reduced_data, cluster_labels)
    except ModuleNotFoundError:
        print("ML_Modules not found. Skipping evaluation.")
    except AttributeError:
        print("evaluate_clusterer function not found in ML_Modules. Skipping evaluation.")


if __name__ == "__main__":
    main()
