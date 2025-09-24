from data.load_data import load_adult_data
from preprocessing.preprocess import preprocess_data
from preprocessing.visualisation import plot_correlation_matrix
from modeling.train_model import choose_and_train_model
from evaluation.evaluate import evaluate_model, print_most_important_feature
from utils.helpers import print_separator
from preprocessing.visualisation import plot_encoded_sex_income_relationship
from modeling.train_model import print_most_important_feature_per_class_logreg


#ucitavanje podataka iz CSV fajlova
#uklanjanje duplikata i anomalija
#podjela na training i test skup
""" X_train, X_test, y_train, y_test = load_adult_data(
    train_path="data/adult_train.csv",
    test_path="data/adult_test.csv"
) """
#Generiše i prikazuje matricu korelacije za sve numeričke kolone
#plot_correlation_matrix(X_train)

#obrada nedostajucih vrednosti i anomalija
#normalizacija numerickih kolona
#enkodiranje
X_train_enc, X_test_enc, y_train_enc, y_test_enc, feature_names = preprocess_data(
    train_path="data/adult_train.csv", test_path="data/adult_test.csv"
)
#plot_encoded_sex_income_relationship(X_train_enc, y_train_enc, feature_names)
print_separator()
print("Podaci su preprocesirani i spremni za treniranje.")

# Treniranje modela i vaznost parametara
# unutar ove fje ja zovem i optimizaciju hiperparametara modela LightGBM
model, y_pred = choose_and_train_model(X_train_enc, X_test_enc, y_train_enc, y_test_enc, model_type="LightGBM", feature_names=feature_names)
print("Model je treniran.")

# Evaluacija
print_separator()
evaluate_model(model_name=model, y_true=y_test_enc, y_pred=y_pred, labels=['<=50K', '>50K'])

print_separator()
print_most_important_feature(model=model, feature_names=feature_names)
