import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_excel("Resources\ML470_S1_HR_Data_Practice.xlsx")
df = df.select_dtypes(include=['number'])


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

