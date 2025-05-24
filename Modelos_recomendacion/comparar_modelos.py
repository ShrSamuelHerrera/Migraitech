
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix

# Cargar los datos originales y preprocesar
print("\n📦 Cargando y preparando datos...")
df = pd.read_excel("test_modificado_agrupacion_gravedad.xlsx", sheet_name="Sheet1")

df["Eficacia"] = (df["Eficacia"] - df["Eficacia"].min()) / (df["Eficacia"].max() - df["Eficacia"].min())
df["genero"] = df["genero"].map({"Femenino": 0, "Masculino": 1})

# Codificar etiquetas
label_encoder = joblib.load("label_encoder_farmacos.pkl")
y = label_encoder.transform(df["farmaco"])

# Definir características
features = ["edad", "IMC_categorica", "genero", "Eficacia", "Adherencia", "Gravedad Total"]
features += [col for col in df.columns if col.startswith("Comorbilidad_") or col.startswith("EA_")]
X = df[features]

# Dividir datos (misma semilla)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargar modelos y scaler
print("\n🔍 Cargando modelos...")
modelo_logistic = joblib.load("modelo_logistic_reg.pkl")
modelo_rf = joblib.load("modelo_random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# Aplicar scaler solo al modelo de regresión logística
X_test_scaled = scaler.transform(X_test)

# Evaluar ambos modelos
def evaluar(nombre, modelo, X, y_test, clases):
    y_pred = modelo.predict(X)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"\n📊 {nombre}:")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print("Recall Promedio:", round(recall, 4))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=clases))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Ejecutar comparaciones
evaluar("Regresión Logística", modelo_logistic, X_test_scaled, y_test, label_encoder.classes_)
evaluar("Random Forest", modelo_rf, X_test, y_test, label_encoder.classes_)
