import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("Resources/ML470_S2_Practice_HR_Data.xlsx")

le_salary = LabelEncoder()
le_dept = LabelEncoder()

df['salary_enc'] = le_salary.fit_transform(df['salary'])
df['Department_enc'] = le_dept.fit_transform(df['Department'])

df.drop(columns=['salary', 'Department'], inplace=True)

features = df.drop(columns=['left'])

corr_matrix = features.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".4f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Employee Features")
plt.show()
