import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

salary_encoder = LabelEncoder()
dpt_encoder = LabelEncoder()

df['salary.enc'] = salary_encoder.fit_transform(df['salary'])
df['Department.enc'] = dpt_encoder.fit_transform(df['Department'])

df.drop(columns=['salary', 'Department'], inplace=True)

x = df.drop(columns='left')
y = df['left']

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
        param_grid = param_grid,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1
    )

grid.fit(xtrain, ytrain)
best_model = grid.best_estimator_

ypred = best_model.predict(xtest)

print("Confusion Matrix")
print(confusion_matrix(ytest, ypred))
print("===================")
print()

print("Classification Report:")
print(classification_report(ytest, ypred, digits=3, zero_division=0))

print("===================")
print(f"accuracy: {accuracy_score(ytest, ypred):.3f}")
print(f"recall: {recall_score(ytest, ypred, zero_division=0):.3f}")
print(f"f1-score: {f1_score(ytest, ypred, zero_division=0):.3f}")
print(f"precision: {precision_score(ytest, ypred, zero_division=0):.3f}")