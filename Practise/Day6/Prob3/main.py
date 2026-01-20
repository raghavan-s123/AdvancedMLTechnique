import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

x = df.drop(columns={'price_range'})
y = df['price_range'].values

scaler = StandardScaler()

x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

rfk = RepeatedKFold(
        10,
        1,
        random_state=42
    )


model = AdaBoostClassifier(random_state=42)
ypred = cross_val_predict(model, x_scaled, y, cv=rfk)

con = confusion_matrix(y, ypred)

print("Confusion Matrix")
print(con)
print("===================")
print()

print("Classification Report:")
print(classification_report(y, ypred, digits=2))

print("===================")

acc = accuracy_score(y, ypred)
rec = recall_score(y, ypred, average="weighted")
f1 = f1_score(y, ypred, average="weighted")
prec = precision_score(y, ypred, average="weighted")

print(f"accuracy: {acc:.3f}")
print(f"recall: {rec:.3f}")
print(f"f1-score: {f1:.3f}")
print(f"precision: {prec:.3f}")
