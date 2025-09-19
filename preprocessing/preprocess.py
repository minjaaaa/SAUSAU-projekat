from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocesiranje podataka:
    - Kodiranje kategorijskih kolona sa OneHotEncoder
    - Kodiranje ciljne varijable sa LabelEncoder
    """
    
    #PRE ENKODIRANJA OBRADI ANOMALIJE

    # podela kolona na numeričke i kategoričke
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # inicijalizacija OneHotEncoder-a za kategoričke kolone
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # 1. Kodiranje kategoričkih kolona
    # Fit na trening skupu i transformacija
    X_train_cat_encoded = enc.fit_transform(X_train[cat_cols])
    # Samo transformacija na test skupu
    X_test_cat_encoded = enc.transform(X_test[cat_cols])

    # 2. Spajanje enkodiranih kategorija sa numeričkim kolonama
    # Koristimo numpy.hstack za horizontalno spajanje
    X_train_encoded = np.hstack([X_train[num_cols], X_train_cat_encoded])
    X_test_encoded = np.hstack([X_test[num_cols], X_test_cat_encoded])

    # 3. Kodiranje ciljne varijable (LabelEncoder)
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded

def detect_outliers(df):
    sns.boxplot(x=df['age'])
    plt.title("Boxplot za age")
    plt.show()

if __name__ == "__main__":
    pass