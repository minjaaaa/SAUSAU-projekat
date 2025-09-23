import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform

def hyperParam(X_train, y_train, model_type):
    """
    Pronalazi optimalne hiperparametre za LogisticRegression model
    koristeći RandomizedSearchCV.

    Parametri:
    X_train (np.array): Enkodirani trening skup
    y_train (np.array): Enkodirana ciljna varijabla trening skupa.

    Vraća:
    tuple: Torka sa najboljim parametrima.
    """
    # Definišite distribucije verovatnoće za parametre
    # loguniform se koristi za C jer je to tipično u logaritamskoj skali
    param_grid = [
        # Za 'l1' penal, koristite 'liblinear' solver
        {'penalty': ['l1'], 'solver': ['liblinear'], 'C': [0.01, 0.1, 1, 10, 100]},
        
        # Za 'l2' penal, možete koristiti i 'liblinear' i 'lbfgs'
        {'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'C': [0.01, 0.1, 1, 10, 100]},
    ]
    param_distributions = {
        'n_estimators': randint(50, 300), # Broj stabala
        'learning_rate': uniform(0.01, 0.1), # Brzina učenja
        'num_leaves': randint(20, 60), # Broj listova
        'class_weight': ['balanced', None] # Balansiranje klasa
    }

    # Instancirajte LogisticRegression model
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    lgbm = LGBMClassifier(random_state=42)
    
    # Inicijalizujte RandomizedSearchCV
    rand_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_distributions,
        n_iter=100, # Broj nasumičnih kombinacija koje treba isprobati
        cv=3, #cross-validation
        scoring='f1', 
        verbose=1,
        # Ovo sprečava grešku kada se kombinuje l1 i lbfgs
        error_score='raise'
    )

    # Pokrenite pretragu
    rand_search.fit(X_train, y_train)

    print("Najbolji parametri pronađeni sa RandomizedSearchCV:")
    print(rand_search.best_params_)


    if model_type == "RandomForest":
        # Vraćanje parametara pojedinačno
        best_C = rand_search.best_params_['C']
        best_solver = rand_search.best_params_['solver']
        best_penalty = rand_search.best_params_['penalty']
        return best_C, best_solver, best_penalty
    else:
        return rand_search.best_estimator_