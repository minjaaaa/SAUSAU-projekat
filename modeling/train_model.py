from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score
from utils.helpers import print_separator
from modeling.hyperparameters import hyperParam

def choose_and_train_model(X_train, X_test, y_train, y_test, model_type="RandomForest", feature_names=None):
    
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "KNeighbors":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "LogisticRegression":
        best_C, best_solver, best_penalty= hyperParam(X_train, y_train)

        model = LogisticRegression(
        C=best_C,
        penalty=best_penalty,
        random_state=42,
        max_iter=1000,
        solver=best_solver
        )
    elif model_type == "LightGBM":
        # class_weight='balanced' je ključan za neravnotežu klasa
        model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    elif model_type == "XGBoost":
        # scale_pos_weight se koristi za neravnotežu klasa
        # Njegova vrednost je odnos broja uzoraka negativne i pozitivne klase
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError("Nepodržan tip modela.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prscr = precision_score(y_test, y_pred)
    
    print(f"\n--- Rezultati za {model_type} model ---")
    print(f"Tačnost: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Preciznost: {prscr:.4f}")

    print_separator()
    print("\nModel je obučen sa najboljim parametrima!")
    print_separator()

    # Prikaz važnosti obeležja (samo za RandomForest i LogisticRegression)
    """ if model_type == "RandomForest":
        importances = model.feature_importances_
        if feature_names:
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.4f}")
    elif model_type == "LogisticRegression":
        # Koeficijenti se koriste kao indikator važnosti
        importances = model.coef_[0]
        if feature_names:

            #feature_importance_list = list(zip(feature_names, importances))
            #feature_importance_list.sort(key=lambda x: x[1], reverse=True) #sortirani prikaz

            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.4f}") """
    
    #vaznosti parametara za modele RandomForest, LightGBM, i XGBoost
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.4f}")
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.4f}")

    return model