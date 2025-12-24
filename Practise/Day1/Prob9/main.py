import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from ML_Modules import auc_roc

df = pd.read_excel("Resources/ML470_S1_HR_Data_Practice.xlsx")

df_encoded = pd.get_dummies(df, columns=["Department", "salary"])

X = df_encoded.drop("left", axis=1)
y = df_encoded["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

model = GaussianNB()
model.fit(X_train, y_train)

auc_roc(model, X_test, y_test)
