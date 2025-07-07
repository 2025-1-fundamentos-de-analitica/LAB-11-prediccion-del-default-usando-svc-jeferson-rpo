# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

def pregunta11():
    import pandas as pd
    import pickle
    import gzip
    import os
    import json

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        balanced_accuracy_score,
        confusion_matrix,
    )

    # Cargar datos
    with open("files/grading/x_train.pkl", "rb") as f:
        x_train = pickle.load(f)
    with open("files/grading/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open("files/grading/x_test.pkl", "rb") as f:
        x_test = pickle.load(f)
    with open("files/grading/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    # Columnas
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical = [col for col in x_train.columns if col not in categorical]

    # Preprocesamiento
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical),
    ])

    # Pipeline
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("pca", PCA(n_components=None)),
        ("select", SelectKBest(score_func=f_classif)),
        ("clf", SVC(kernel="rbf", class_weight="balanced")),
    ])

    # Búsqueda de hiperparámetros
    param_grid = {
        "select__k": [15, 20, "all"],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"]
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    # Guardar modelo
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)



    # Función para métricas
def reparar_metrics_json():
    import pickle, gzip, json
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
    )
    import os

    # Cargar modelo
    with gzip.open("files/models/model.pkl.gz", "rb") as f:
        model = pickle.load(f)

    # Cargar datos
    with open("files/grading/x_train.pkl", "rb") as f:
        x_train = pickle.load(f)
    with open("files/grading/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open("files/grading/x_test.pkl", "rb") as f:
        x_test = pickle.load(f)
    with open("files/grading/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    def compute_metrics(x, y, name):
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        return [
            {
                "type": "metrics",
                "dataset": name,
                "precision": precision_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1_score": f1_score(y, y_pred),
            },
            {
                "type": "cm_matrix",
                "dataset": name,
                "true_0": {
                    "predicted_0": int(cm[0][0]),
                    "predicted_1": int(cm[0][1]),
                },
                "true_1": {
                    "predicted_0": int(cm[1][0]),
                    "predicted_1": int(cm[1][1]),
                },
            },
        ]

    # Generar resultados en orden correcto
    train_metrics, train_cm = compute_metrics(x_train, y_train, "train")
    test_metrics, test_cm = compute_metrics(x_test, y_test, "test")

    resultados = [train_metrics, test_metrics, train_cm, test_cm]

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for linea in resultados:
            json.dump(linea, f)
            f.write("\n")

if __name__ == "__main__":
    reparar_metrics_json()