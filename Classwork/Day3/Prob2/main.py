import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("Resources/ML470_S3_Diabetes_Data_Preprocessed_Concept.xlsx")

X = df[["Fasting blood", "bmi", "FamilyHistory", "HbA1c", "age"]]
y = df["target"]

# Train model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
model.fit(X, y)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Non-Diabetic", "Diabetic"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Visualization for Diabetes Prediction")
plt.show()
