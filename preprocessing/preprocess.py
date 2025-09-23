import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer # Importujte SimpleImputer

# Vaša lista sa nazivima kolona
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

def preprocess_data(train_path="data/adult_train.csv", test_path="data/adult_test.csv"):
    """
    Učitava, čisti i preprocesira podatke za mašinsko učenje.
    
    Vraća:
        X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, all_feature_names
    """
    # 1. Učitavanje i početno čišćenje
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)

    for col in combined_df.select_dtypes(include='object').columns:
        combined_df[col] = combined_df[col].astype(str).str.strip().str.replace('.', '', regex=False)
        combined_df[col].replace('?', np.nan, inplace=True)
    
    # Uklanjanje kolone 'fnlwgt' (nije relevantna za predikciju)
    combined_df = combined_df.drop('fnlwgt', axis=1)

    # Odvajanje obeležja i ciljne varijable
    X = combined_df.drop('income', axis=1)
    y = combined_df['income']
    
    # 2. Uklanjanje redova sa NaN vrednostima u ciljnoj varijabli i uklanjanje retkih klasa
    valid_mask = y.notna()
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()

    class_counts = y.value_counts()
    single_member_classes = class_counts[class_counts < 2].index
    
    if not single_member_classes.empty:
        valid_mask_split = ~y.isin(single_member_classes)
        X = X[valid_mask_split].copy()
        y = y[valid_mask_split].copy()
    
    # 3. Stratifikovana podela na trening i test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,
        random_state=42, 
        stratify=y
    )

    # 4. Odvajanje numeričkih i kategoričkih kolona
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # 5. Imputacija nedostajućih vrednosti
    for col in cat_cols:
        most_frequent = X_train[col].mode()[0]
        X_train[col].fillna(most_frequent, inplace=True)
        X_test[col].fillna(most_frequent, inplace=True)
        
    for col in num_cols:
        # Koristimo SimpleImputer za numericke
        imputer = SimpleImputer(strategy='mean')
        X_train[col] = imputer.fit_transform(X_train[[col]])
        X_test[col] = imputer.transform(X_test[[col]])

    # 6. Obrada anomalija (outliera) u numeričkim kolonama
    # Isključujemo 'capital-gain' i 'capital-loss' zbog prirode podataka
    outlier_cols = [col for col in num_cols if col not in ['capital-gain', 'capital-loss']]
    for col in outlier_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)

    # 7. Normalizacija numeričkih kolona
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])

    # 8. Kodiranje kategoričkih kolona
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_encoded = enc.fit_transform(X_train[cat_cols])
    X_test_cat_encoded = enc.transform(X_test[cat_cols])

    # 9. Spajanje obeležja
    encoded_feature_names = enc.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_feature_names)
    X_train_encoded = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_encoded = np.hstack([X_test_num_scaled, X_test_cat_encoded])

    # 10. Kodiranje ciljne varijable
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, all_feature_names