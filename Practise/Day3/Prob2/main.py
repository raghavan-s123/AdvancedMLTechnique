import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("Resources/ML470_S3_HR_Data_Practice.xlsx")

salary_encoder = LabelEncoder()
dept_encoder = LabelEncoder()

df["salary.enc"] = salary_encoder.fit_transform(df["salary"])
df["Department.enc"] = dept_encoder.fit_transform(df["Department"])

df_encoded = df.drop(columns=["salary", "Department"])

corr_matrix = df_encoded.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title("Correlation Heatmap of Employee Features")
plt.show()

numeric_features = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary.enc",
    "Department.enc",
    "left"
]

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_encoded[numeric_features])
plt.title("Box Plot Analysis of Numeric Employee Features")
plt.xlabel("Features")
plt.show()
