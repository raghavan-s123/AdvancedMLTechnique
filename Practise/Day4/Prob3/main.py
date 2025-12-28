import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

df.drop(columns=['salary', 'Department'], inplace=True)

x = df.drop(columns='average_monthly_hours')
y = df['average_monthly_hours']

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)

grid_param = {
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
        estimator=model,
        param_grid=grid_param,
        cv=5,
        scoring='neg_mean_squared_error'
    )

grid.fit(xtrain, ytrain)

best_model = grid.best_estimator_

ypred = best_model.predict(xtest)

cv_mse = -grid.best_score_
test_mse = mean_squared_error(ytest, ypred)
var = y.var(ddof=0)


print(f"Cross-validated MSE (after tuning): {cv_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Best Hyperparameters: {grid.best_params_}")
print(f"Variance of target: {var:.3f}")