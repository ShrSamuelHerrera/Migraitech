import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar los datos
df = pd.read_excel("test_modificado_agrupacion_gravedad.xlsx")

# Normalizar 'Eficacia'
df["Eficacia"] = (df["Eficacia"] - df["Eficacia"].min()) / (df["Eficacia"].max() - df["Eficacia"].min())

# Codificar género
df["genero"] = df["genero"].map({"Femenino": 0, "Masculino": 1})

# Codificar farmaco
label_encoder = LabelEncoder()
df["farmaco"] = label_encoder.fit_transform(df["farmaco"])
farmacos_nombres = label_encoder.classes_

# Reducir casos de Amitriptilina a 150
amitrip_code = label_encoder.transform(["Amitriptilina"])[0]
df_ami = df[df["farmaco"] == amitrip_code]
df_ami_reducido = resample(df_ami, replace=False, n_samples=150, random_state=42)
df_rest = df[df["farmaco"] != amitrip_code]
df_balanceado = pd.concat([df_ami_reducido, df_rest])

# Definir características
features = ["edad", "IMC_categorica", "genero", "Eficacia", "Adherencia", "Gravedad Total"]
features += [col for col in df_balanceado.columns if col.startswith("Comorbilidad_") or col.startswith("EA_")]

X = df_balanceado[features]
y = df_balanceado["farmaco"]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000, class_weight='balanced')
modelo.fit(X_train, y_train)

# Guardar modelo, codificador y scaler
joblib.dump(modelo, "modelo_logistic_reg.pkl")
joblib.dump(label_encoder, "label_encoder_farmacos.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluación
def evaluar(modelo, X_test, y_test, clases):
    y_pred = modelo.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=clases))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=clases,
                yticklabels=clases)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

evaluar(modelo, X_test, y_test, farmacos_nombres)
