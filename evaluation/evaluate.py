from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluacija modela: accuracy i confusion matrix.
    """
    y_pred = model.predict(X_test)
    #acc = accuracy_score(y_test, y_pred)
    #precision = precision_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    #print(f"Accuracy: {acc:.4f}")
    #print(f"Precision: {precision:.4f}")
    #print(f"F1 score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
