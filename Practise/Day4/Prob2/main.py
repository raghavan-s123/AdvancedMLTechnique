import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

df.drop(columns=['salary', 'Department'], inplace=True)

x = df.drop(columns='average_monthly_hours')
y = df['average_monthly_hours']

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)

mse_scores = cross_val_score(
        model,
        x_scaled,
        y,
        cv=5,
        scoring='neg_mean_squared_error'
    )

mean_mse = -mse_scores.mean()
print(f"Cross-validated MSE: {mean_mse}")

model.fit(x_scaled, y)

ypred = model.predict(x_scaled)

print(f"Predictions: {ypred}")

