import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Resources/ML470_S1_HR_Data_Practice.xlsx")

data = pd.crosstab(df['salary'], df['left'])

data.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Salary Level')
plt.ylabel('Number of Employees')
plt.legend(['0', '1'])
plt.xticks(rotation=0)
plt.show()