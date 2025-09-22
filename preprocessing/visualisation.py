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

def visualise(train_path="data/adult_train.csv"):
    df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, skipinitialspace=True) #ucitavanje podataka
    #brise se whitespace
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    plt.figure(figsize=(6,4))
    sns.countplot(x='income', data=df)
    plt.title("Distribucija income")
    plt.show()

    #missing(df)
    #korelacija(df)
    #plot_correlation_matrix(df)

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

if __name__ == "__main__":
    
    visualise(train_path="data/adult_train.csv")