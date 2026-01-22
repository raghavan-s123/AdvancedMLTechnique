import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print(df.head())
print()

df.info()
print()

scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

eps = 2

min_samples = [3, 4, 5]

for sample in min_samples:
    
    model = DBSCAN(eps, sample)
    label = model.fit_predict(df)
    
    label = label + 2

    unique, counts = np.unique(label, return_counts=True)
    
    cluster = list(zip(unique, counts))
    
    print(f"eps= {eps} | min_samples=  {sample} | obtained clustering:  {cluster}")

print()

final = DBSCAN(2, 3)
final_labels = final.fit_predict(df)

cluster_counts = pd.Series(final_labels, name="cluster").value_counts()
print(cluster_counts)
print()


if len(set(final_labels)) > 1:
    sil = silhouette_score(df, final_labels)
    cal = calinski_harabasz_score(df, final_labels)
    dav = davies_bouldin_score(df, final_labels)
    
    print(f"The average silhouette_score is: {sil:.2f}\n")
    print(f"Calinski-Harabasz Index: {cal:.2f}\n")
    print(f"Davies-Bouldin Index: {dav:.2f}")


