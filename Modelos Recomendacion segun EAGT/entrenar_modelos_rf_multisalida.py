import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

# Cargar dataset
DATASET_PATH = "test_modificado_agrupacion_gravedad.xlsx"
df = pd.read_excel(DATASET_PATH)

# Eliminar columnas que solo contienen ceros
df = df.drop(columns=df.columns[(df == 0).all()])

# Redondear variables numéricas para mejorar generalización
df["peso"] = df["peso"].round(1)
df["IMC"] = df["IMC"].round(1)
df["Adherencia"] = df["Adherencia"].round(1)
df["Eficacia"] = df["Eficacia"].round(1)

# Clasificar Eficacia como binaria: Alta vs No Alta

def clasificar_eficacia(valor):
    return "Alta" if valor >= 7.5 else "No Alta"

df["Eficacia_clasificada"] = df["Eficacia"].apply(clasificar_eficacia)

# Codificar genero
if df["genero"].dtype == object:
    df["genero"] = df["genero"].map({"Femenino": 0, "Masculino": 1})

# Separar variables
features = df.drop(columns=["Eficacia", "Adherencia", "Gravedad Total", "Eficacia_clasificada"])
y_eficacia_clas = df["Eficacia_clasificada"]
y_aderencia = df["Adherencia"]
y_gravedad = df["Gravedad Total"]

# One-hot encoding para farmaco
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(features[["farmaco"]])
X_otros = features.drop(columns=["farmaco"]).reset_index(drop=True)
X_final = pd.concat([X_otros, pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["farmaco"]))], axis=1)
X_final.columns = [col.replace(" ", "_") for col in X_final.columns]

# Train/test split
X_train, X_test, y_train_efi_clas, y_test_efi_clas = train_test_split(X_final, y_eficacia_clas, test_size=0.2, random_state=42)
_, _, y_train_ade, y_test_ade = train_test_split(X_final, y_aderencia, test_size=0.2, random_state=42)
_, _, y_train_gra, y_test_gra = train_test_split(X_final, y_gravedad, test_size=0.2, random_state=42)

# Aplicar SMOTE para balancear clases
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_efi_clas_bal = smote.fit_resample(X_train, y_train_efi_clas)



# Codificar etiquetas de eficacia
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_efi_clas_bal_enc = le.fit_transform(y_train_efi_clas_bal)
y_test_efi_clas_enc = le.transform(y_test_efi_clas)

# Entrenar modelos
model_efi_clas = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_efi_clas.fit(X_train_bal, y_train_efi_clas_bal_enc)

model_ade = RandomForestRegressor().fit(X_train, y_train_ade)
model_gra = RandomForestRegressor().fit(X_train, y_train_gra)

# Guardar modelos y encoder
joblib.dump(model_efi_clas, "modelo_clas_eficacia.pkl")
joblib.dump(model_ade, "modelo_rf_aderencia.pkl")
joblib.dump(model_gra, "modelo_rf_gravedad.pkl")
joblib.dump(encoder, "encoder_farmacos.pkl")

print("\n✅ Modelos entrenados y guardados como .pkl")

# Evaluar modelos
print("📊 Evaluación para Eficacia (clasificación):")
y_pred_efi_clas = model_efi_clas.predict(X_test)
print(classification_report(y_test_efi_clas, le.inverse_transform(y_pred_efi_clas)))

# Mostrar matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_efi_clas, le.inverse_transform(y_pred_efi_clas), labels=["Alta", "No Alta"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Alta", "No Alta"], yticklabels=["Alta", "No Alta"])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Eficacia")
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

def evaluar_modelo(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print(f"\n📊 Evaluación para {nombre}:")
    print("  R²:", round(r2_score(y_test, y_pred), 4))
    print("  MAE:", round(mean_absolute_error(y_test, y_pred), 4))
    print("  RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

evaluar_modelo("Adherencia", model_ade, X_test, y_test_ade)
ConfusionMatrixDisplay.from_predictions(y_test_ade.round(), model_ade.predict(X_test).round(), cmap="Blues")
plt.title("Matriz de Confusión - Adherencia")
plt.show()
evaluar_modelo("Gravedad", model_gra, X_test, y_test_gra)
ConfusionMatrixDisplay.from_predictions(y_test_gra.round(), model_gra.predict(X_test).round(), cmap="Purples")
plt.title("Matriz de Confusión - Gravedad")
plt.show()
