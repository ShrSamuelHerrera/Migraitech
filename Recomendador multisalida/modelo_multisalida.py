import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Cargar dataset
DATASET_PATH = "test_modificado_agrupacion_gravedad.xlsx"
df = pd.read_excel(DATASET_PATH)

# Eliminar columnas que solo contienen ceros
df = df.drop(columns=df.columns[(df == 0).all()])

# Redondear variables numéricas
df["peso"] = df["peso"].round(1)
df["IMC"] = df["IMC"].round(1)
df["Eficacia"] = df["Eficacia"].round(1)

# Codificar género
df["genero"] = df["genero"].map({"Femenino": 0, "Masculino": 1})

# Submuestreo más agresivo: reducir casos con eficacia < 15 en Amitriptilina
casos_amitriptilina = df[(df["farmaco"] == "Amitriptilina") & (df["Eficacia"] < 15)]
df_reducido_amitriptilina = casos_amitriptilina.sample(frac=0.15, random_state=42)  # solo 15% se conserva
otros_datos = df[~((df["farmaco"] == "Amitriptilina") & (df["Eficacia"] < 15))]
df = pd.concat([otros_datos, df_reducido_amitriptilina], ignore_index=True)

# Agrupar eficacia en clases personalizadas
def clasificar_eficacia(valor):
    if valor < 2.5:
        return "Muy baja"
    elif valor < 6:
        return "Baja"
    elif valor < 15:
        return "Media"
    else:
        return "Alta"

df["Eficacia_clas"] = df["Eficacia"].apply(clasificar_eficacia)

# Preparar features y etiquetas para eficacia
features_eficacia = df[["edad", "genero", "peso", "altura", "IMC", "farmaco"]]
y_eficacia = df["Eficacia_clas"]

# One-hot encoding para fármaco
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(features_eficacia[["farmaco"]])
X_otros = features_eficacia.drop(columns=["farmaco"]).reset_index(drop=True)
X_final = pd.concat([
    X_otros,
    pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["farmaco"]))
], axis=1)
X_final.columns = [col.replace(" ", "_") for col in X_final.columns]

# Codificar etiquetas a números
le = LabelEncoder()
y_encoded = le.fit_transform(y_eficacia)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42
)

# Balanceo: SMOTE + submuestreo suave de la clase mayoritaria
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Submuestreo suave de la clase mayoritaria
from collections import Counter
counts = Counter(y_smote)
maj_class, maj_count = counts.most_common(1)[0]
total_other = sum(c for k, c in counts.items() if k != maj_class)
target_maj = min(int(total_other * 0.5), maj_count)
rus = RandomUnderSampler(sampling_strategy={maj_class: target_maj}, random_state=42)
X_train_bal, y_train_bal = rus.fit_resample(X_smote, y_smote)

# Entrenar modelo con datos balanceados
model_efi = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_efi.fit(X_train_bal, y_train_bal)

# Guardar modelo, encoder y label encoder
joblib.dump(model_efi, "modelo_eficacia.pkl")
joblib.dump(encoder, "encoder_farmacos.pkl")
joblib.dump(le, "labelencoder_eficacia.pkl")

print("\n✅ Modelo de eficacia entrenado y guardado como .pkl")

# Evaluación
print("\n📊 Evaluación para Eficacia:")
y_pred = model_efi.predict(X_test)
print(classification_report(
    le.inverse_transform(y_test), le.inverse_transform(y_pred)
))
