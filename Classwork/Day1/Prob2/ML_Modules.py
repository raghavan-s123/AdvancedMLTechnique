from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

def evaluate_classifier(ytest, ypred):
    
    con = confusion_matrix(ytest, ypred)
    print("Confusion Matrix")
    print(con)
    print("===================")
    
    acc = accuracy_score(ytest, ypred)
    rec = recall_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    prec = precision_score(ytest, ypred)
    
    print(f"accuracy: {acc:.3f}")
    print(f"recall: {rec:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {prec:.3f}")
    