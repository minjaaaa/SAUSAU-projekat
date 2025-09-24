from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model_name, y_true, y_pred, labels=None):
    print(f"\n--- Izveštaj o performansama modela: {model_name} ---")

    # Matrica konfuzije
    cm = confusion_matrix(y_true, y_pred)
    
    # Detaljan klasifikacioni izveštaj
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

    # Ukupne metrike
    accuracy = accuracy_score(y_true, y_pred)
    weighted_precision = precision_score(y_true, y_pred)
    weighted_recall = recall_score(y_true, y_pred)
    weighted_f1_score = f1_score(y_true, y_pred)

    print(f"\nModel tip: {model_name}")
    print(f"\nUkupna tačnost (Accuracy): {accuracy:.4f}")
    print(f"Ukupna preciznost (Precision - weighted): {weighted_precision:.4f}")
    print(f"Ukupan odziv (Recall - weighted): {weighted_recall:.4f}")
    print(f"Ukupan F1-Score (weighted): {weighted_f1_score:.4f}")
    
    print("\nDetaljan klasifikacioni izveštaj:")
    print(report)
    
    print("\nMatrica konfuzije:")
    print(cm)

def print_most_important_feature(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        print("Model nema 'feature_importances_' atribut.")
        return

    importances = model.feature_importances_
    grouped_importances = {} #recnik kategorija : vaznost

    for i, name in enumerate(feature_names): #ovdje je enkodovana vrednost
        # izdvajanje originalnog naziva kategorije
        original_col_name = name.split('_')[0]
        
        # ako grupa ne postoji, napravi je
        if original_col_name not in grouped_importances:
            grouped_importances[original_col_name] = []
        
        # dodajem vaznosti
        grouped_importances[original_col_name].append((name, importances[i]))

    print(f"\n--- Najvažnije obeležje po kategoriji za model '{model.__class__.__name__}' ---")
    
    
    for category, features in grouped_importances.items():
        if not features:
            continue
        
        # maksimizujem po vaznosti
        most_important = max(features, key=lambda item: item[1])
        
        print(f"Najvažnije iz kategorije '{category}': {most_important[0]} (Vrednost: {most_important[1]:.4f})")
