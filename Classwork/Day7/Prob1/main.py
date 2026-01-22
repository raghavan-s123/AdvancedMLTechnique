import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# 1. Load and Explore Data
# -----------------------------
data = pd.read_csv("Resources/ML470_S7_MallCustomers_Data_Concept.csv")

X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

print("Initial Data:")
print(X.head())

# -----------------------------
# 2. Outlier Detection (IQR)
# -----------------------------
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
print("\nOutlier Count:")
print(outliers.sum())

# -----------------------------
# 3. Outlier Treatment (Capping)
# -----------------------------
for col in X.columns:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    X[col] = np.clip(X[col], lower, upper)

# -----------------------------
# 4. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Box Plot (After Scaling)
# -----------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(X_scaled, labels=X.columns)
plt.title("Box Plot of Scaled Customer Features")
plt.ylabel("Scaled Value")
plt.show()

# -----------------------------
# 6. Dendrogram (Ward Linkage)
# -----------------------------
linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title("Customer Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# -----------------------------
# 7. Dendrogram with Cut Line
# -----------------------------
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.axhline(y=8, color='red', linestyle='--')
plt.title("Customer Dendrogram with Distance Threshold")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()
