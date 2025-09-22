from data.load_data import load_adult_data
from preprocessing.preprocess import preprocess_data
from modeling.train_model import train_model
from evaluation.evaluate import evaluate_model
from utils.helpers import print_separator

#ucitavanje podataka iz CSV fajlova
#uklanjanje duplikata i anomalija
#podjela na training i test skup
X_train, X_test, y_train, y_test = load_adult_data(
    train_path="data/adult_train.csv",
    test_path="data/adult_test.csv"
)

#obrada nedostajucih vrednosti i anomalija
#enkodiranje
X_train_enc, X_test_enc, y_train_enc, y_test_enc, feature_names = preprocess_data(
    X_train, X_test, y_train, y_test
)

print_separator()
print("Podaci su preprocesirani i spremni za treniranje.")

# Treniranje modela i vaznost parametara
model = train_model(X_train_enc, y_train_enc, feature_names)
print("Model je treniran.")

# Evaluacija
print_separator()
evaluate_model(model, X_test_enc, y_test_enc)
print_separator()
