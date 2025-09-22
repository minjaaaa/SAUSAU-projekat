from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def choose_and_train_model(X_train, X_test, y_train, y_test, model_type="RandomForest", feature_names=None):
    """
    Trenira i evaluira odabrani model, i prikazuje važnost obeležja.
    """
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "KNeighbors":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError("Nepodržan tip modela.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- Rezultati za {model_type} model ---")
    print(f"Tačnost: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Prikaz važnosti obeležja (samo za RandomForest i LogisticRegression)
    if model_type == "RandomForest":
        importances = model.feature_importances_
        if feature_names:
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.4f}")
    elif model_type == "LogisticRegression":
        # Koeficijenti se koriste kao indikator važnosti
        importances = model.coef_[0]
        if feature_names:
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.4f}")

    return model