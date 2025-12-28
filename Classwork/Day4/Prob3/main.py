import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


file = input()

df = pd.read_csv(os.path.join(sys.path[0], file))
print(df.head())
print()

X = df[['bmi', 'age', 'insulin', 'FamilyHistory', 'bp']]
y = df['Fasting blood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_

print(f"Best Hyperparameters: {best_params}")

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-cv_scores)
print(f"Cross-Validation RMSE Scores: {rmse_scores}")

mean_rmse = rmse_scores.mean()
print(f"Mean RMSE: {mean_rmse}")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {test_rmse}")

std_label = y.std()
print(f"Standard Deviation of Label: {std_label}")

if test_rmse <= std_label:
    print("The model's RMSE is within the standard deviation, indicating good performance.")
else:
    print("The model's RMSE exceeds the standard deviation, suggesting room for improvement.")
