import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def hyperParam(X_train, y_train):
    param_grid = [
        # validne kombinacije
        {'penalty': ['l1'], 'solver': ['liblinear'], 'C': [0.01, 0.1, 1, 10, 100]},
        
        {'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'C': [0.01, 0.1, 1, 10, 100]},
    ]

    logreg = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Najbolji parametri: {grid_search.best_params_}")
    
    best_C = grid_search.best_params_['C']
    best_solver = grid_search.best_params_['solver']
    best_penalty = grid_search.best_params_['penalty']

    # Vraćanje parametara pojedinačno
    return best_C, best_solver, best_penalty