from data.load_data import load_adult_data
from preprocessing.preprocess import preprocess_data
from preprocessing.visualisation import plot_correlation_matrix
from modeling.train_model import choose_and_train_model
from evaluation.evaluate import evaluate_model
from utils.helpers import print_separator


#ucitavanje podataka iz CSV fajlova
#uklanjanje duplikata i anomalija
#podjela na training i test skup
X_train, X_test, y_train, y_test = load_adult_data(
    train_path="data/adult_train.csv",
    test_path="data/adult_test.csv"
)
#Generiše i prikazuje matricu korelacije za sve numeričke kolone
plot_correlation_matrix(X_train)
#obrada nedostajucih vrednosti i anomalija
#normalizacija numerickih kolona
#enkodiranje
X_train_enc, X_test_enc, y_train_enc, y_test_enc, feature_names = preprocess_data(
    X_train, X_test, y_train, y_test
)

print_separator()
print("Podaci su preprocesirani i spremni za treniranje.")

# Treniranje modela i vaznost parametara
model = choose_and_train_model(X_train_enc, X_test_enc, y_train_enc, y_test_enc, model_type="RandomForest", feature_names=None)
print("Model je treniran.")

# Evaluacija
print_separator()
#evaluate_model(model, X_test_enc, y_test_enc)
print_separator()
