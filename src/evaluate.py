import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os
import seaborn as sns


def eval_model(filename):
    df = pd.read_csv(os.path.join('./data/processed', filename))
    label =  pd.read_csv(os.path.join('./data/processed', "label.csv"))
    label = label['Churn']
        # Leemos el modelo entrenado para usarlo
    package = './models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df
    y_test = label
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)

# Validación desde el inicio
def main():
    df = eval_model('Data_Customer_Churn_train.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()