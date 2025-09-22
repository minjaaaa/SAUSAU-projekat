import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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
    
    #print("Tipovi podataka nakon ucitavanja:")
    #print(train_df.dtypes)

    #print(train_df['age'].dtype) int64
    # Pre-spajanje, obradi '?' vrednosti u obe tabele i zameni ih sa NaN
    # Ovo osigurava da Pandas zadrži numerički tip podatka
    # Brisanje whitespace-ova
    for col in train_df.select_dtypes(include='object').columns:
        train_df[col] = train_df[col].str.strip()
    for col in test_df.select_dtypes(include='object').columns:
        test_df[col] = test_df[col].str.strip()

    # KONVERTOVANJE SVIH NUMERICKIH KOLONA U BROJEVE PRE SPAJANJA
    for col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').astype('Int64')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype('Int64')

    # Spajanje trening i test skupa za ujednačenu obradu (pre podele)
    # Ovo se radi kako bi se osiguralo da su sve operacije primenjene konzistentno na celom datasetu
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Tip podatka 'age' nakon transformacije: {combined_df['age'].dtype}") #OVDJE AGE POSTAJE OBJECT

    # Uklanjanje duplikata iz celog dataseta
    combined_df.drop_duplicates(inplace=True)
    
    # Normalizacija ciljne kolone pre podele: ukloni whitespace i tačku
    combined_df['income'] = combined_df['income'].str.strip().str.replace('.', '', regex=False)

    # Odvajanje obeležja (X) i ciljne varijable (y)
    X = combined_df.drop('income', axis=1)
    y = combined_df['income']

    # Izbacivanje redova gde je ciljna kolona NaN
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    # **********************************************
    # Stratifikovana podela na trening i test set, imam neuravnotezen odnos target-a
    # **********************************************
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, #70% podataka koristim za trening
        random_state=42, 
        stratify=y # osigurava ravnomernu raspodelu klasa.
    )
    # Prikaz raspodele klasa u trening i test skupu
    #print("Raspodela klasa u y_train:")
    #print(y_train.value_counts(normalize=True))
    #print("\nRaspodela klasa u y_test:")
    #print(y_test.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test

def detect_duplicate(df):
    """
    Proverava duplikate u train setu i uklanja ih.
    Vraća očišćen DataFrame bez duplikata.
    """

    duplicates = df.duplicated()
    print("Postoje li duplikati?", duplicates.any())

    print("Broj redova pre uklanjanja duplikata:", df.shape[0])

    num_duplicates = df.duplicated().sum()
    print("Broj duplikata:", num_duplicates)

    # uklanjanje duplikata
    df_clean = df.drop_duplicates()

    # provera posle
    print("Broj redova posle uklanjanja duplikata:", df_clean.shape[0])
    return df_clean

if __name__ == "__main__":
    
    load_adult_data(train_path="data/adult_train.csv", test_path="data/adult_test.csv")


