import pandas as pd
import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

file = input()
df = pd.read_csv(os.path.join(sys.path[0], file))

cv = CountVectorizer()

x = cv.fit_transform(df['HealthText'])
y = df['Outcome']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

mnb = MultinomialNB()
mnb.fit(xtrain, ytrain)

ypred = mnb.predict(xtest)


new_sample = ["Age group: Senior | BMI status: Overweight | Glucose category: Very High Glucose Level"]
new_vec = cv.transform(new_sample)
prediction = mnb.predict(new_vec)


print("Prediction: ",prediction[0])

