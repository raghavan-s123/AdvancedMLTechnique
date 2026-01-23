import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import warnings

warnings.filterwarnings("ignore")

def load_file(file_path):
    
    ext = file_path.split(".")[-1].lower()
    if ext in ["xlsx", "xls"]:
        return pd.read_excel(file_path)
    elif ext == "csv":
        return pd.read_csv(file_path)
    else:
        print("Error: Unsupported file format. Please use CSV or Excel.")
        sys.exit(1)

def main():
    
    filename = input("Enter your dataset filename (CSV or Excel): ").strip()
    file_path = os.path.join(sys.path[0], filename)


    try:
        df = load_file(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

   
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        print("Error: No numeric columns found.")
        sys.exit(1)

    print("\n========== NUMERIC COLUMNS ==========")
    print(df_numeric.columns.tolist())

    X = df_numeric.values

    
    print("\n========== SILHOUETTE SCORES ==========")
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=10)
        labels = model.fit_predict(X)
        score = round(silhouette_score(X, labels), 3)
        print(f"k = {k}: Silhouette Score = {score}")

    
    print("\n========== FINAL CLUSTER MODEL (k=2) ==========")
    final_k = 2
    kmeans = KMeans(n_clusters=final_k, random_state=10)
    cluster_labels = kmeans.fit_predict(X)
    print("Cluster Labels:")
    print(cluster_labels)

    
    print("\n========== CLUSTER EVALUATION ==========")
    print("Cluster Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette_score(X, cluster_labels):.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, cluster_labels):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X, cluster_labels):.4f}")

if __name__ == "__main__":
    main()
