
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar los datos desde el Excel actualizado
df = pd.read_excel("test_modificado_agrupacion_gravedad.xlsx", sheet_name="Sheet1")

# Normalizar la variable 'Eficacia'
df["Eficacia"] = (df["Eficacia"] - df["Eficacia"].min()) / (df["Eficacia"].max() - df["Eficacia"].min())

# Codificar 'genero'
df["genero"] = df["genero"].map({"Femenino": 0, "Masculino": 1})

# Codificar 'farmaco'
label_encoder_farmaco = LabelEncoder()
df["farmaco"] = label_encoder_farmaco.fit_transform(df["farmaco"])
farmacos_nombres = label_encoder_farmaco.classes_

# Seleccionar características
features = ["edad", "IMC_categorica", "genero", "Eficacia", "Adherencia", "Gravedad Total"]
features += [col for col in df.columns if col.startswith("Comorbilidad_") or col.startswith("EA_")]

X = df[features]
y = df["farmaco"]

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)

# Guardar el modelo y el codificador
joblib.dump(model, "modelo_random_forest.pkl")
joblib.dump(label_encoder_farmaco, "label_encoder_farmacos_rf.pkl")

# Evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test, nombres_clases):
    y_pred = modelo.predict(X_test)
    print("\n📊 Métricas de Evaluación:")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall Promedio: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score Promedio: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\n📝 Reporte de Clasificación:")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=nombres_clases))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

evaluar_modelo(model, X_test, y_test, farmacos_nombres)

# Mostrar importancia de variables
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🔍 Top 10 variables más importantes para la predicción:")
print(importances.head(10))
