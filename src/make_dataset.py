#Importamos las librerías

import pandas as pd
import numpy as np
import os

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('./data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):

    data = df.replace(" ",np.nan) # replace("El dato que quieres reemplazar","El dato por el cual lo quieres reemplazar")
    data.dropna(inplace = True); # identique a todos los valores nulos "NaN", lo puedo eliminar con la funcion dropna()
    data["TotalCharges"] = data["TotalCharges"].astype("float")
    data["SeniorCitizen"] = data["SeniorCitizen"].astype("str")

    # Guardar todas las variables categoricas en un solo lugar
    cat_cols = data[['gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod']]

    num_cols = data[['tenure', 'MonthlyCharges', 'TotalCharges']]

    # Generar variables para las dos columnas que omiti de mi mapeo de variables cualitativas y cuantitativas
    #id_customer = data["customerID"]
    label = data["Churn"]

    # La variable target categorica solo puede ser transformada con el target encoding
    label = label.apply(lambda x: 1 if x == "Yes" else 0) # Yes - 1, No -0
    label.to_csv(os.path.join('./data/processed/', 'label.csv'))

    data.drop("tenure",inplace = True, axis = 1) #eliminamos la variable tenencia
    data.drop("Churn",inplace = True, axis = 1) #eliminamos la variable target de la data original
 #   data.drop("customerID",inplace = True, axis = 1) #eliminamos la variable id de cliente

    data = pd.get_dummies(data = data) #transformamos las variables categóricas a numéricas

#   data_original = pd.concat([data, label,id_customer], axis=1)

    return data

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('./data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('Data_Customer_Churn.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male',
       'SeniorCitizen_0', 'SeniorCitizen_1', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'], 'Data_Customer_Churn_train.csv')
    

if __name__ == '__main__':
    print("Proyecto Final - Python Essentials for ETL")
    print("===========================================")
    print("")
    print("----------------------------------")
    print("         Integrantes:")
    print("----------------------------------")
    print("     - Alexander Rodriguez")
    print("     - José Magallón")
    print("----------------------------------")
 
    main()