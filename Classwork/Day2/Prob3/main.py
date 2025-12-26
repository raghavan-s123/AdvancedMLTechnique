import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

x = df[['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']]
y = df['Outcome']

zero_cols = ['Glucose', 'BMI']

mask = (x[zero_cols] != 0).all(axis=1)

x = x[mask]
y = y[mask]

for col in x.columns:
    
    Q1 = x[col].quantile(0.25)
    Q3 = x[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    x[col] = x[col].clip(lower, upper)

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size = 0.2, random_state=42)

svc = SVC()

param_grid = {
    'C' : [0.1, 1, 10, 100],
    'gamma' : ['scale', 'auto', 0.01, 0.1, 1],
    'kernel' : ['rbf']
}

grid = GridSearchCV(
        estimator = svc,
        param_grid   = param_grid,
        cv = 5,
        scoring='accuracy',
        n_jobs=-1
    )
grid.fit(xtrain, ytrain)
best_model = grid.best_estimator_

ypred = best_model.predict(xtest)

print("Confusion Matrix")
print(confusion_matrix(ytest, ypred))
print("===================")
print()

print("Classification Report:")
print(classification_report(ytest, ypred))
print("===================")

print(f"accuracy: {accuracy_score(ytest, ypred):.3f}")
print(f"recall: {recall_score(ytest, ypred):.3f}")
print(f"f1-score: {f1_score(ytest, ypred):.3f}")
print(f"precision: {precision_score(ytest, ypred):.3f}")



