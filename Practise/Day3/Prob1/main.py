import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print("First 5 rows of the dataset:")
print(df.head())
print()

print("Number of samples in the data:")
print(len(df))
print()

print("Data types of each column:")
print(df.dtypes)
print()

df.drop(columns = ['salary', 'Department'], inplace=True)
x = df.drop(columns=['left'])
y = df['left']

print("Feature columns:")
print(list(x.columns))
print()

print("Statistical summary of numeric columns:")
print(df.describe())

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


dtc = DecisionTreeClassifier(random_state=42, max_depth=4)
dtc.fit(xtrain, ytrain)

ypred = dtc.predict(xtest)

print(f"Model Accuracy: {accuracy_score(ytest, ypred):.1f}")

print("Classification Report:")
print(classification_report(ytest, ypred))
