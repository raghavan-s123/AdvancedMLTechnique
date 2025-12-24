from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def auc_roc(classifier, X_test, y_test):
   
    y_prob = classifier.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC–AUC Score: {auc_score:.4f}")

    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
