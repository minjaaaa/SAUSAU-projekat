import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

    # 2. Obrada anomalija u svim numeričkim kolonama
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols: #granice se racunaju samo na trening skupu
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound) #ali granice primenjujem i na test skup
        #model sada generalizuje, i neka precica tipa age>80 target>=50K
    
    # 3. Podela kolona na numeričke i kategoričke
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Kodiranje kategoričkih kolona
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_encoded = enc.fit_transform(X_train[cat_cols])
    X_test_cat_encoded = enc.transform(X_test[cat_cols])

    # 5. Spajanje enkodiranih kategorija sa numeričkim kolonama
    encoded_feature_names = enc.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_feature_names)

    X_train_encoded = np.hstack([X_train[num_cols], X_train_cat_encoded])
    X_test_encoded = np.hstack([X_test[num_cols], X_test_cat_encoded])

    # 6. Kodiranje ciljne varijable
    le_target = LabelEncoder()
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
    detect_outliers(train_path="data/adult_train.csv")