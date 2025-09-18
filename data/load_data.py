import pandas as pd

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

def load_adult_data(train_path="data/adult_train.csv", test_path="data/adult_test.csv"):
    """
    Učitaj train i test CSV fajlove bez zaglavlja i dodaj imena kolona.
    Izbaci redove gde je ciljna kolona NaN i normalizuj income vrednosti.
    """
    # Učitavanje CSV fajlova
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)

    # Trim whitespace iz string kolona
    for col in train_df.select_dtypes(include='object').columns:
        train_df[col] = train_df[col].str.strip()
    for col in test_df.select_dtypes(include='object').columns:
        test_df[col] = test_df[col].str.strip()

    # Split data i target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Izbacivanje redova gde je ciljna kolona NaN
    train_notna = y_train.notna()
    X_train = X_train[train_notna]
    y_train = y_train[train_notna]

    test_notna = y_test.notna()
    X_test = X_test[test_notna]
    y_test = y_test[test_notna]

    # Normalizacija ciljne kolone: ukloni whitespace i tačku
    y_train = y_train.str.strip().str.replace('.', '', regex=False)
    y_test = y_test.str.strip().str.replace('.', '', regex=False)

    return X_train, X_test, y_train, y_test
