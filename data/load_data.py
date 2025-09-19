import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    # ucitavanje CSV fajlova
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)

    # brisem whitespace-ove
    for col in train_df.select_dtypes(include='object').columns:
        train_df[col] = train_df[col].str.strip()
    for col in test_df.select_dtypes(include='object').columns:
        test_df[col] = test_df[col].str.strip()

    #uklanjanje duplikata iz train skupa
    train_df = detect_duplicate(train_df)
    #detect_outliers(train_df)
    sns.countplot(x='income', data=train_df)

    #URADI NESTO SA ANOMALIJAMA PRIJE NEGO PODJELIS SKUP !!!!!!!!!!!
    #i sa nedostajucim vrednostima

    
    
    X_train = train_df.iloc[:, :-1] #preuzimam sve kolone osim poslednje-target izlaza
    y_train = train_df.iloc[:, -1] #izdvajam izlaz - model ne sme da ga vidi u toku treniranja
    X_test = test_df.iloc[:, :-1] #isto i za test skup
    y_test = test_df.iloc[:, -1]

    # Izbacivanje redova gde je ciljna kolona NaN
    train_notna = y_train.notna() #vraca masku True/False gdje izlaz nije/jeste Nan
    X_train = X_train[train_notna] 
    y_train = y_train[train_notna] #nema Nan vrednosti

    test_notna = y_test.notna() #postoji samo 1 Nan u y_test, ali baca error svakako
    X_test = X_test[test_notna]
    y_test = y_test[test_notna]

    # normalizacija ciljne kolone: ukloni whitespace i tačku
    y_train = y_train.str.strip().str.replace('.', '', regex=False)
    y_test = y_test.str.strip().str.replace('.', '', regex=False)

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

def detect_outliers(df):
    sns.boxplot(x=df['age'])
    plt.title("Boxplot za age")
    plt.show()

if __name__ == "__main__":
    
    load_adult_data(train_path="data/adult_train.csv", test_path="data/adult_test.csv")


