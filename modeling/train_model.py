from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def train_model(X_train, y_train, feature_names): #napravi dodatni fajl, gdje treniras jos modela
    """
    Treniranje RandomForest modela i prikaz vaznosti 
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42) #tacnost 0.8517
    #model = KNeighborsClassifier(n_neighbors=5) #tacnost 0.7768
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

    return model
