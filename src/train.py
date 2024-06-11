import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('./data/processed', filename))
    label =  pd.read_csv(os.path.join('./data/processed', "label.csv"))
    label = label['Churn']

    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size = 0.3)

    xg_boost = XGBClassifier()
    apply_classifier(xg_boost,X_train, X_test, y_train, y_test)

    print('Modelo exportado correctamente en la carpeta models')


def apply_classifier(clf,xTrain,yTrain):

    clf.fit(xTrain, yTrain) #Entrenamiento del modelo

    # Guardamos el modelo entrenado para usarlo en produccion
    package = './models/best_model.pkl'
    pickle.dump(clf, open(package, 'wb'))

# Entrenamiento completo
def main():
    read_file_csv('Data_Customer_Churn_train.csv')

if __name__ == "__main__":
    main()