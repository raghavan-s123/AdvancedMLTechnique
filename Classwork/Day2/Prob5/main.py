import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# -----------------------------------
# 1. Generate Moons Dataset
# -----------------------------------
X, y = make_moons(
    n_samples=300,
    noise=0.2,
    random_state=42
)

# -----------------------------------
# Visualization 1: Raw Moons Dataset
# -----------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Moons Dataset (Non-Linear Structure)")
plt.grid(True)
plt.show()

# -----------------------------------
# 2. Train Linear SVM
# -----------------------------------
linear_svm = SVC(kernel='linear')
linear_svm.fit(X, y)

# Mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Z_linear = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_linear = Z_linear.reshape(xx.shape)

# -----------------------------------
# Visualization 2: Linear SVM Boundary
# -----------------------------------
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z_linear, cmap='coolwarm', alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear SVM Decision Boundary")
plt.grid(True)
plt.show()

# -----------------------------------
# 3. Train RBF Kernel SVM
# -----------------------------------
rbf_svm = SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X, y)

Z_rbf = rbf_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

# -----------------------------------
# Visualization 3: RBF Kernel SVM Boundary
# -----------------------------------
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z_rbf, cmap='coolwarm', alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Non-Linear SVM (RBF Kernel) Decision Boundary")
plt.grid(True)
plt.show()
