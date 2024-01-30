from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix
)



def get_score(clf, X, y, train=True):

    pred = clf.predict(X)
    
    metrics = {
        'accuracy': accuracy_score(y, pred),
        'clf_report': classification_report(y, pred),
        'cm': confusion_matrix(y, pred)
    }

    return pred, metrics


def print_score(y_true, y_pred, train=True, add_cr=False, add_cm=False):

    if train:
        print('================================================')
        print("Train Result:")
    else:
        print('================================================')
        print("Test Result:")
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")

    if add_cr:
        clf_report = classification_report(y_true, y_pred)
        print("_______________________________________________")
        print(f"Classification report:\n{clf_report}")

    if add_cm:
        cm = confusion_matrix(y_true, y_pred)
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {cm}\n")