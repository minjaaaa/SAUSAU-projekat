from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train): #napravi dodatni fajl, gdje treniras jos modela
    """
    Treniranje RandomForest modela.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42) #tacnost 0.8517
    #model = KNeighborsClassifier(n_neighbors=5) #tacnost 0.7768
    model.fit(X_train, y_train)
    return model
