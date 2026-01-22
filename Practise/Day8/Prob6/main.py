import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score 

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[col] = df[col].clip(lower, upper)
    
df.drop(columns='Detergents_Paper', inplace=True)

scaler = StandardScaler()
scaled = scaler.fit_transform(df)

dbscan = DBSCAN(2, 5)
labels = dbscan.fit_predict(scaled)

sil = silhouette_score(scaled, labels)
chi = calinski_harabasz_score(scaled, labels)
dav = davies_bouldin_score(scaled, labels)

print(f"The average silhouette_score is: {sil:.2f}")
print(f"Calinski-Harabasz Index: {chi:.2f}")
print(f"Davies-Bouldin Index: {dav:.2f}")

    