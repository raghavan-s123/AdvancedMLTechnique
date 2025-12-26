import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# -------------------------------
# 1. Create dataset
# -------------------------------
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=42
)

# -------------------------------
# 2. Train Linear SVM
# -------------------------------
model = SVC(kernel='linear')
model.fit(X, y)

# -------------------------------
# 3. Create mesh grid
# -------------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# -------------------------------
# 4. Plot data points
# -------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0][:, 0], X[y == 0][:, 1],
    color='red', label='Apples (Class 0)'
)

plt.scatter(
    X[y == 1][:, 0], X[y == 1][:, 1],
    color='orange', label='Oranges (Class 1)'
)

# -------------------------------
# 5. Plot decision boundary
# -------------------------------
plt.contour(
    xx, yy, Z,
    levels=[0],
    colors='black',
    linewidths=2
)

# -------------------------------
# 6. Plot margins
# -------------------------------
plt.contour(
    xx, yy, Z,
    levels=[-1, 1],
    colors='black',
    linestyles='dashed'
)

# -------------------------------
# 7. Highlight support vectors
# -------------------------------
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=120,
    facecolors='none',
    edgecolors='black',
    label='Support Vectors'
)

# -------------------------------
# 8. Labels and legend
# -------------------------------
plt.xlabel("Fruit Size")
plt.ylabel("Sweetness")
plt.title("Linear SVM Fruit Classification")
plt.legend()
plt.grid(True)

plt.show()
