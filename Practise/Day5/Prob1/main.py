import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print()
print("--- Outlier Assessment ---")

for col in df.columns:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outlier = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outlier} outliers")
    df[col] = df[col].clip(lower, upper)
    
x = df.drop(columns='Purchase Likelihood')
y = df['Purchase Likelihood']

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

split = int(0.8 * len(x_scaled))

xtrain = x_scaled[:split]
xtest = x_scaled[split:]
ytrain = y[:split]
ytest = y[split:]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)
acc = accuracy_score(ytest, ypred) * 100;

print()
print("==============================")
print(f"Model Accuracy: {round(acc, 2)} %")
print("==============================")



