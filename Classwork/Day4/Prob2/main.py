import pandas as pd
import os
import sys
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

x = df.drop(columns='target').values
y = df['target'].values

model = DecisionTreeClassifier(random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "max_depth" : [2, 3, 4, 5, 6],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3]
}

grid = GridSearchCV(
        estimator=model,
        param_grid = param_grid,
        cv=skf,
        scoring="accuracy",
        n_jobs= -1
    )

grid.fit(x, y)

best_params = grid.best_params_
best_score = grid.best_score_

print(f"Best Hyperparameters: {best_params}")
print(f"Best Stratified CV Accuracy: {best_score:.3f}")