import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read Excel file
df = pd.read_excel("Resources/ML470_S2_Diabetes_Data_Concept.xlsx")   # change filename if needed

# Select only required columns
cols = ["BloodPressure", "BMI", "Age", "FamilyHistory", "HbA1c"]
num_df = df[cols]

# ==============================
# Before Outlier Treatment
# ==============================
plt.figure(figsize=(10, 6))
sns.boxplot(data=num_df)
plt.title("Before Outlier Treatment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==============================
# IQR Outlier Treatment
# ==============================
treated_df = num_df.copy()

for col in treated_df.columns:
    Q1 = treated_df[col].quantile(0.25)
    Q3 = treated_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    treated_df[col] = treated_df[col].clip(lower, upper)

# ==============================
# After Outlier Treatment
# ==============================
plt.figure(figsize=(10, 6))
sns.boxplot(data=treated_df)
plt.title("After Outlier Treatment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


