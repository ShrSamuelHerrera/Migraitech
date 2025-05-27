
# Importación de librerías necesarias
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar el dataset desde un archivo Excel
DATASET_PATH = "datos_pacientes_entrenamiento.xlsx"
df = pd.read_excel(DATASET_PATH)

# Eliminar columnas que contienen únicamente ceros
df = df.drop(columns=df.columns[(df == 0).all()])

# Redondear variables numéricas para mejorar la generalización
df["peso"] = df["peso"].round(1)
df["IMC"] = df["IMC"].round(1)
df["Adherencia"] = df["Adherencia"].round(1)
df["Eficacia"] = df["Eficacia"].round(1)

# Crear una nueva variable categórica binaria para clasificar la eficacia
# Si es mayor o igual al percentil 75 se clasifica como "Alta", en caso contrario como "No Alta"
p75 = df["Eficacia"].quantile(0.75)
df["Eficacia_clasificada"] = df["Eficacia"].apply(lambda x: "Alta" if x >= p75 else "No Alta")

# Codificar el género en valores numéricos (0 para Femenino, 1 para Masculino)
if df["genero"].dtype == object:
    df["genero"] = df["genero"].map({"Femenino": 0, "Masculino": 1})

# Separar las variables predictoras y las etiquetas
features = df.drop(columns=["Eficacia", "Adherencia", "Eficacia_clasificada"])
y_eficacia_clas = df["Eficacia_clasificada"]
y_aderencia = df["Adherencia"]

# Aplicar one-hot encoding a la columna "farmaco"
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(features[["farmaco"]])
X_otros = features.drop(columns=["farmaco"]).reset_index(drop=True)
X_final = pd.concat([X_otros, pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["farmaco"]))], axis=1)
X_final.columns = [col.replace(" ", "_") for col in X_final.columns]

# Dividir los datos en conjuntos de entrenamiento y prueba para ambos modelos
X_train, X_test, y_train_efi_clas, y_test_efi_clas = train_test_split(X_final, y_eficacia_clas, test_size=0.2, random_state=42)
X_train_ade, X_test_ade, y_train_ade, y_test_ade = train_test_split(X_final, y_aderencia, test_size=0.2, random_state=42)

# Balancear las clases de eficacia utilizando SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_efi_clas_bal = smote.fit_resample(X_train, y_train_efi_clas)

# Codificar las etiquetas de eficacia como valores numéricos
le = LabelEncoder()
y_train_efi_clas_bal_enc = le.fit_transform(y_train_efi_clas_bal)
y_test_efi_clas_enc = le.transform(y_test_efi_clas)

# Entrenar modelo de clasificación para eficacia con XGBoost
model_efi_clas = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_efi_clas.fit(X_train_bal, y_train_efi_clas_bal_enc)

# Entrenar modelo de regresión para adherencia con Random Forest
model_ade = RandomForestRegressor().fit(X_train_ade, y_train_ade)

# Guardar modelos y codificador en archivos .pkl
joblib.dump(model_efi_clas, "modelo_clas_eficacia.pkl")
joblib.dump(model_ade, "modelo_rf_aderencia.pkl")
joblib.dump(encoder, "encoder_farmacos.pkl")

print("\n✅ Modelos entrenados y guardados como .pkl")

# Evaluación del modelo de clasificación (eficacia)
print("\n📊 Evaluación para Eficacia (clasificación):")
y_pred_efi_clas = model_efi_clas.predict(X_test)
print(classification_report(y_test_efi_clas, le.inverse_transform(y_pred_efi_clas)))

# Mostrar matriz de confusión para la clasificación de eficacia
cm = confusion_matrix(y_test_efi_clas, le.inverse_transform(y_pred_efi_clas), labels=["Alta", "No Alta"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Alta", "No Alta"], yticklabels=["Alta", "No Alta"])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Eficacia")
plt.show()

# Función para evaluar modelo de regresión (adherencia)
def evaluar_modelo(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print(f"\n📊 Evaluación para {nombre}:")
    print("  R²:", round(r2_score(y_test, y_pred), 4))
    print("  MAE:", round(mean_absolute_error(y_test, y_pred), 4))
    print("  RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

# Evaluar el modelo de adherencia y mostrar matriz de confusión redondeada
evaluar_modelo("Adherencia", model_ade, X_test_ade, y_test_ade)
ConfusionMatrixDisplay.from_predictions(y_test_ade.round(), model_ade.predict(X_test_ade).round(), cmap="Blues")
plt.title("Matriz de Confusión - Adherencia")
plt.show()
