import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

def visualise(train_path="data/adult_test.csv"):
    
    sns.set(style="whitegrid")

    df = pd.read_csv(train_path)

    # Pregled osnovnih statistika
    print("\nStatistika dataset-a:")
    print(df.describe())

    # Informacije o kolonama (tipovi, null vrednosti)
    print("\nInfo o dataset-u:")
    print(df.info())

    # -------------------------------
    # 3️⃣ Histogram numeričkih kolona
    # -------------------------------
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(8,5))
        plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Distribucija: {col}")
        plt.xlabel(col)
        plt.ylabel("Broj")
        plt.show()

    # -------------------------------
    # Boxplot za numeričke kolone (po kategorijama)
    # -------------------------------
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    """ # po prvoj kategorijskoj koloni
    if numeric_cols and categorical_cols:
        plt.figure(figsize=(9,6))
        sns.boxplot(x=categorical_cols[0], y=numeric_cols[0], data=df)
        plt.xticks(rotation=45)
        plt.title(f"{numeric_cols[0]} po {categorical_cols[0]}")
        plt.show() """

    # -------------------------------
    # 5️⃣ Countplot za kategorijske kolone
    # -------------------------------
    for col in categorical_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(x=col, data=df)
        plt.xticks(rotation=45)
        plt.title(f"Brojnost po kategoriji: {col}")
        plt.show()

    # -------------------------------
    # 6️⃣ Scatter plot za dve numeričke kolone
    # -------------------------------
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
        plt.show()


def missing(df):
    # Prebrojavanje frekvencije svih vrednosti u koloni 'workclass'
    workclass_counts = df['workclass'].value_counts()

    # Prikazivanje broja ponavljanja znaka '?'
    print(f"Broj ponavljanja '?' u koloni 'workclass': {workclass_counts.get('?', 0)}") #oko 5.6%, previse da bi se izbacilo

def korelacija(df):
    #df.replace('?', pd.NA, inplace=True)
    #df.dropna(inplace=True)
    plt.figure(figsize=(15, 8)) # Povećava veličinu grafika za bolju čitljivost
    sns.boxplot(x='workclass', y='age', data=df)

    plt.title('Distribucija godina po radnom statusu')
    plt.xlabel('Radni status (Workclass)')
    plt.ylabel('Godine (Age)')
    plt.xticks(rotation=45) # Rotiranje x-ose radi bolje čitljivosti
    plt.grid(True, linestyle='--', alpha=0.6) # Dodavanje mreže
    plt.show()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    

    # Prikaz heatmap-e
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelaciona matrica")
    plt.show()

def plot_encoded_sex_income_relationship(X_encoded, y_encoded, all_feature_names):
    """
    Prikazuje zavisnost enkodiranog pola i zarade.
    """
    # Kreiranje privremenog DataFrame-a
    temp_df = pd.DataFrame(X_encoded, columns=all_feature_names)
    temp_df['income'] = y_encoded
    
    # Pronađite enkodirane kolone za 'sex'
    sex_cols = [col for col in all_feature_names if 'sex' in col]
    
    # Proverite da li su kolone pronađene
    if not sex_cols:
        print("Nije moguće pronaći enkodirane kolone za 'sex'.")
        return
    
    # Kreirajte novu kolonu 'sex' sa originalnim vrednostima
    # Na osnovu 1/0 enkodiranja, 0 je obično muško, a 1 žensko
    if len(sex_cols) == 2:
        temp_df['sex'] = temp_df[sex_cols[0]] # Pretpostavimo da je prva kolona 'sex_Male'
        temp_df['sex'] = temp_df['sex'].map({1: 'Male', 0: 'Female'})
    else:
        # Alternativno, ako je korišten drop_first=True
        temp_df['sex'] = temp_df[sex_cols[0]]
        temp_df['sex'] = temp_df['sex'].map({1: 'Female', 0: 'Male'})
    
    plt.figure(figsize=(8, 6))
    
    # Kreiranje bar-dijagrama
    sns.countplot(x='sex', hue='income', data=temp_df)
    
    plt.title('Zavisnost zarade od pola (enkodirani podaci)')
    plt.xlabel('Pol')
    plt.ylabel('Broj instanci')
    plt.show()

if __name__ == "__main__":
    
    df = pd.read_csv("data/adult_test.csv")
    