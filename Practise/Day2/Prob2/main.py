import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Resources/ML470_S2_Practice_HR_Data.xlsx")

num_cols = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 6))
sns.boxplot(data=num_cols, color='red')

plt.xticks(rotation=45)
plt.title("Boxplots for Outlier Detection")
plt.show()
