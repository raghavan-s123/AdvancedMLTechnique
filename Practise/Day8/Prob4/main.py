import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

for col in df.columns:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[col] = df[col].clip(lower, upper)

df.drop(columns='Detergents_Paper',inplace=True)

scaler = StandardScaler()

scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

eps = 2.0

for value in [3, 4, 5]:
    
    dbscan = DBSCAN(eps, value)
    
    label = dbscan.fit_predict(scaled)
    
    label = label + 2
    
    unique, counts = np.unique(label, return_counts=True)
    
    results = list(zip(unique, counts))
    
    print(f"eps = {int(eps)} | min_samples = {value} | obtained clustering: {results}")
    
