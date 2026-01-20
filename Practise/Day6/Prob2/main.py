import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# -------------------------------------------------
# Read CSV filename
# -------------------------------------------------

# Load dataset
df = pd.read_excel("Resources\ML470_S6_Mobile_Data_Practice.xlsx")

# -------------------------------------------------
# Separate input features and target
# -------------------------------------------------
X = df.drop(columns=["price_range"])

# -------------------------------------------------
# Visualization 1: Outlier Assessment (Boxplots)
# -------------------------------------------------
plt.figure(figsize=(15, 8))
X.boxplot()
plt.xticks(rotation=90)
plt.title("Outlier Assessment Using Boxplots")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Visualization 2: Feature Correlation Heatmap
# -------------------------------------------------
corr_matrix = X.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Feature Correlation Analysis Using Heatmap")
plt.tight_layout()
plt.show()
