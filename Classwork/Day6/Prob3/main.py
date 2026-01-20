import pandas as pd
import os
import sys
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score


warnings.simplefilter("ignore")

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

print(df.head())

x = df.drop(columns='price_range.enc')
y = df['price_range.enc']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

dt = DecisionTreeClassifier(max_depth = 2, random_state=42)

model = AdaBoostClassifier(base_estimator=dt, n_estimators=50, learning_rate=1.0, random_state=42)

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

con = confusion_matrix(ytest, ypred)
acc = accuracy_score(ytest, ypred)
rec = recall_score(ytest, ypred, average="weighted")
f1 = f1_score(ytest, ypred, average="weighted")
prec = precision_score(ytest, ypred, average="weighted")

print()
print("Confusion Matrix")
print(con)
print("===================")
print()

print("Classification Report:")
print(classification_report(ytest, ypred, digits=2))
print("===================")

print(f"accuracy: {acc:.3f}")
print(f"recall: {rec:.3f}")
print(f"f1-score: {f1:.3f}")
print(f"precision: {prec:.3f}")



