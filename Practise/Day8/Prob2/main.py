import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 1. Load and Prepare Dataset
# -------------------------------

# Automatically detect CSV file
# Load dataset
df = pd.read_csv("Resources/ML470_S8_Wholesale Customers data_Practice.csv")

# Select numerical columns and round values
num_df = df.select_dtypes(include=np.number).round(2)

print("Numerical Features Used:")
print(num_df.columns.tolist())

# -------------------------------
# 2. Visualization 1: Boxplot
# -------------------------------

def plot_boxplots(data):
    plt.figure(figsize=(12, 6))
    data.boxplot(rot=90)
    plt.title("Boxplot for Outlier Detection")
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

plot_boxplots(num_df)

# -------------------------------
# 3. Outlier Treatment (IQR Capping)
# -------------------------------

def iqr_capping(data):
    capped_data = data.copy()
    for col in capped_data.columns:
        Q1 = capped_data[col].quantile(0.25)
        Q3 = capped_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        capped_data[col] = capped_data[col].clip(lower, upper)
    return capped_data

treated_df = iqr_capping(num_df)

# -------------------------------
# 4. Visualization 2: Correlation Heatmap
# -------------------------------

def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap for Multicollinearity Analysis")
    plt.tight_layout()
    plt.show()

plot_correlation_heatmap(treated_df)
