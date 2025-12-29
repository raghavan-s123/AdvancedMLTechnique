import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score


df = pd.read_excel("Resources/ML470_S5_RetailSales_Data_Practice.xlsx")

target_col = "Purchase Likelihood"


X = df.drop(columns=[target_col])
y = df[target_col]

X_numeric = X.select_dtypes(include=["int64", "float64"])


plt.figure(figsize=(16, 7))
sns.boxplot(data=X_numeric, color="tan", linewidth=1)

plt.title("Comprehensive Box Plot Analysis of Customer Features")
plt.xlabel("Features")
plt.ylabel("Value")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    color="orange",
    linewidth=2,
    label=f"Random Forest (AUC = {auc_score:.2f})"
)

plt.plot([0, 1], [0, 1], linestyle="--", color="blue", label="No Skill")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Purchase Likelihood Model")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
