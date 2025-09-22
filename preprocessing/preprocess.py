import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocesiranje podataka:
    - Obrada nepoznatih vrednosti ('?')
    - Obrada anomalija (outliera) u svim numeričkim kolonama
    - Kodiranje kategorijskih kolona sa OneHotEncoder
    - Kodiranje ciljne varijable sa LabelEncoder
    """
    
     # 1. Obrada nepoznatih vrednosti ('?')
    X_train.replace('?', 'Unknown', inplace=True)
    X_test.replace('?', 'Unknown', inplace=True)

    # 2. Odvajanje numeričkih i kategoričkih kolona
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # 3. Obrada anomalija (outliera) u svim numeričkim kolonama
    for col in num_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)
    
    # ****************************************************
    # 4. Normalizacija numeričkih kolona
    # ****************************************************
    scaler = StandardScaler() #srednja vr. 0, standardna devijacija 1
    
    # Fit na trening skupu i transformacija
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    # Samo transformacija na test skupu
    X_test_num_scaled = scaler.transform(X_test[num_cols]) #ne zelimo da model vidi test skup

    # 5. Kodiranje kategorijskih kolona
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_encoded = enc.fit_transform(X_train[cat_cols])
    X_test_cat_encoded = enc.transform(X_test[cat_cols])

    # 6. Spajanje enkodiranih kategorija sa normalizovanim numeričkim kolonama
    encoded_feature_names = enc.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_feature_names)

    X_train_encoded = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_encoded = np.hstack([X_test_num_scaled, X_test_cat_encoded])

    # 7. Kodiranje ciljne varijable
    le_target = LabelEncoder() #Lbel kodiranj 0, 1, 2..
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, all_feature_names

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

def detect_outliers(train_path="data/adult_train.csv"):
    df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)
    sns.boxplot(x=df['age'])
    plt.title("Boxplot za age")
    plt.show()



if __name__ == "__main__":
    pass