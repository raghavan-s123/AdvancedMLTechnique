# =========================
# 1. Import Required Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# 2. Load Dataset
# =========================
# Change the path if needed
df = pd.read_excel("Resources/ML470_S9_Insurance_Data_Concept.xlsx")

# Features and target
X = df.drop(columns=['weight_condition_n'])
y = df['weight_condition_n']

# =========================
# 3. Train-Test Split (70-30)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# 4. Feature Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. Baseline KNN (No LDA)
# =========================
error_rate = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_test_scaled)
    error_rate.append(np.mean(predictions != y_test))

best_k_baseline = k_values[np.argmin(error_rate)]

# Train final baseline model
knn_baseline = KNeighborsClassifier(n_neighbors=best_k_baseline)
knn_baseline.fit(X_train_scaled, y_train)

y_pred_base = knn_baseline.predict(X_test_scaled)
y_prob_base = knn_baseline.predict_proba(X_test_scaled)

baseline_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_base),
    "Precision": precision_score(y_test, y_pred_base, average='weighted'),
    "Recall": recall_score(y_test, y_pred_base, average='weighted'),
    "F1-Score": f1_score(y_test, y_pred_base, average='weighted'),
    "ROC-AUC": roc_auc_score(y_test, y_prob_base, multi_class='ovr')
}

# =========================
# 6. Apply Linear Discriminant Analysis (LDA)
# =========================
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# =========================
# 7. KNN After LDA
# =========================
error_rate_lda = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_lda, y_train)
    predictions = knn.predict(X_test_lda)
    error_rate_lda.append(np.mean(predictions != y_test))

best_k_lda = k_values[np.argmin(error_rate_lda)]

# Train final LDA-KNN model
knn_lda = KNeighborsClassifier(n_neighbors=best_k_lda)
knn_lda.fit(X_train_lda, y_train)

y_pred_lda = knn_lda.predict(X_test_lda)
y_prob_lda = knn_lda.predict_proba(X_test_lda)

lda_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_lda),
    "Precision": precision_score(y_test, y_pred_lda, average='weighted'),
    "Recall": recall_score(y_test, y_pred_lda, average='weighted'),
    "F1-Score": f1_score(y_test, y_pred_lda, average='weighted'),
    "ROC-AUC": roc_auc_score(y_test, y_prob_lda, multi_class='ovr')
}

# =========================
# 8. Compare Results
# =========================
results_df = pd.DataFrame({
    "Baseline KNN": baseline_metrics,
    "LDA + KNN": lda_metrics
})

print("\nPerformance Comparison:\n")
print(results_df)

# =========================
# 9. Visualization (Bar Chart)
# =========================
results_df.plot(kind='bar')
plt.title("KNN Performance Comparison (With vs Without LDA)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()