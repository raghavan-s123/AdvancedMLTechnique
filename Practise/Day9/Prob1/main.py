import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("Resources/ML470_S9_KCHouse_Data_Practice.xlsx")

df = df.drop(columns=[c for c in df.columns if c.lower() in ["id", "date"]])

numeric_cols = df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(14,6))
df[numeric_cols].boxplot(rot=45)
plt.title("Bar Chart:")
plt.tight_layout()
plt.show()

corr = df[numeric_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
plt.tight_layout()
plt.show()