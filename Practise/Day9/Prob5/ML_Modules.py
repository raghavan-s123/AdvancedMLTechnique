# ML_Modules.py
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clusterer(X, labels):
    """
    Evaluate clustering results using standard metrics.
    """
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    print("\n========== Cluster Evaluation Metrics ==========")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Calinski-Harabasz Index: {ch:.4f}")
    print(f"Davies-Bouldin Index: {db:.4f}")
# ML_Modules.py
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clusterer(X, labels):
    """
    Evaluate clustering results using standard metrics.
    """
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    print("\n========== Cluster Evaluation Metrics ==========")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Calinski-Harabasz Index: {ch:.4f}")
    print(f"Davies-Bouldin Index: {db:.4f}")
