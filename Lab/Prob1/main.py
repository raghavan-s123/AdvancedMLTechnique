import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

def predict_spam():
    
    file_name = input()
    
    file_path = os.path.join(sys.path[0], file_name)

    df = pd.read_csv(file_path)

    
    X = df['Email Content']
    y = df['Label'].map({'Spam' : 1, 'Not Spam' : 0})  # Convert target to binary (1 for Spam, 0 for Not Spam)

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
    
    model = GaussianNB()
    model.fit(X_train.toarray(), y_train)

    y_pred = model.predict(X_test.toarray())

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

predict_spam()
