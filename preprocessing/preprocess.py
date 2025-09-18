from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocesiranje podataka:
    - Kodiranje kategorijskih kolona sa OrdinalEncoder
    - Kodiranje ciljne varijable sa LabelEncoder
    - Nepoznate vrednosti u test setu se kodiraju kao -1
    """
    # Detekcija kategorijskih kolona
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_encoded[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test_encoded[cat_cols] = enc.transform(X_test[cat_cols])

    # Kodiranje ciljne varijable
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded
