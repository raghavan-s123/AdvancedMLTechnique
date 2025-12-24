import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("Resources/ML470_S1_HR_Data_Practice.xlsx")
data = pd.crosstab(df["Department"], df["left"]).plot(kind="bar")
plt.show()

