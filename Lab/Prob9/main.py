# PCA and LDA on Wine Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv("Resources/wine.csv")

# 2. Separate features and target
X = data.drop('Wine', axis=1)
y = data['Wine']

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA (2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Apply LDA (2 components)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 6. Unique classes and colors
classes = y.unique()
colors = ['red', 'green', 'blue']

# 7. PCA Scatter Plot
plt.figure(figsize=(8, 6))
for cls, color in zip(classes, colors):
    plt.scatter(
        X_pca[y == cls, 0],
        X_pca[y == cls, 1],
        label=f'Class {cls}',
        color=color
    )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA (2D Projection) of Wine Dataset')
plt.legend()
plt.show()

# 8. LDA Scatter Plot
plt.figure(figsize=(8, 6))
for cls, color in zip(classes, colors):
    plt.scatter(
        X_lda[y == cls, 0],
        X_lda[y == cls, 1],
        label=f'Class {cls}',
        color=color
    )

plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('LDA (2D Projection) of Wine Dataset')
plt.legend()
plt.show()
