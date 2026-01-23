import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("Resources/ML470_S9_KCHouse_Data_Practice.xlsx")

df = df.drop(columns=[c for c in df.columns if c.lower() in ["id", "date"]])

df["price_range"] = pd.qcut(df["price"], 4, labels=["Low", "Mid", "Upper-Mid", "High"])

le = LabelEncoder()
df["price_range_encoded"] = le.fit_transform(df["price_range"])

X = df.drop(columns=["price", "price_range", "price_range_encoded"])
y = df["price_range_encoded"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

error_rate = []

for k in range(1, 8):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(1 - accuracy_score(y_test, pred))

plt.figure(figsize=(12,6))
plt.plot(range(1,8), error_rate, linestyle="--", marker="o",
         markerfacecolor="red", markeredgecolor="blue")
plt.title("Error Rate vs. K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()