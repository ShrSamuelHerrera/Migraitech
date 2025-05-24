import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar modelos entrenados y encoder
model_efi_clas = joblib.load("modelo_clas_eficacia.pkl")
model_ade = joblib.load("modelo_rf_aderencia.pkl")
model_gra = joblib.load("modelo_rf_gravedad.pkl")
encoder = joblib.load("encoder_farmacos.pkl")

# Cargar nuevos pacientes sin fármaco asignado
nuevos_pacientes = pd.read_excel("nuevos_pacientes.xlsx")  # <- Debes proporcionar este archivo

# Recalcular IMC_categoria si era parte del entrenamiento
if "IMC" in nuevos_pacientes.columns:
    def calcular_imc_categoria(imc):
        if imc < 18.5:
            return 1
        elif imc < 25:
            return 2
        elif imc < 30:
            return 3
        else:
            return 4
    nuevos_pacientes["IMC_categoria"] = nuevos_pacientes["IMC"].apply(calcular_imc_categoria)

nuevos_pacientes["peso"] = nuevos_pacientes["peso"].round(1)
nuevos_pacientes["IMC"] = nuevos_pacientes["IMC"].round(1)

if nuevos_pacientes["genero"].dtype == object:
    nuevos_pacientes["genero"] = nuevos_pacientes["genero"].map({"Femenino": 0, "Masculino": 1})

# Lista de fármacos usados en el entrenamiento
farmacos = encoder.categories_[0]

# Generar predicciones por paciente y fármaco
recomendaciones = []
for idx, paciente in nuevos_pacientes.iterrows():
    filas_farmacos = []
    for farmaco in farmacos:
        paciente_mod = paciente.copy()
        paciente_mod["farmaco"] = farmaco
        paciente_df = pd.DataFrame([paciente_mod])

        X_encoded = encoder.transform(paciente_df[["farmaco"]])
        X_otros = paciente_df.drop(columns=["farmaco"]).reset_index(drop=True)
        X_final = pd.concat([X_otros, pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["farmaco"]))], axis=1)

        # Añadir columnas faltantes con valor 0 para coincidir con entrenamiento
        columnas_modelo = model_efi_clas.feature_names_in_
        faltantes = [col for col in columnas_modelo if col not in X_final.columns]
        if faltantes:
            nuevos = pd.DataFrame(0, index=np.arange(len(X_final)), columns=faltantes)
            X_final = pd.concat([X_final, nuevos], axis=1)

        # Reordenar columnas exactamente como en entrenamiento
        X_final = X_final[columnas_modelo]
        X_final.columns = [col.replace(" ", "_") for col in X_final.columns]

        # Obtener probabilidad de eficacia alta
        eficacia_proba = model_efi_clas.predict_proba(X_final)[0][1]  # Probabilidad de clase 'Alta'
        adherencia = model_ade.predict(X_final)[0]
        gravedad = model_gra.predict(X_final)[0]

        # Score normalizado a 0-100
        raw_score = eficacia_proba * 0.4 + adherencia * 0.3 - gravedad * 0.3
        max_score = 0.3 + 0.5  # gravedad mínima = 0
        score = max(0, min((raw_score / max_score) * 100, 100))

        filas_farmacos.append({
            "paciente_id": paciente["id"],
            "farmaco": farmaco,
            "eficacia": round(eficacia_proba, 3),
            "adherencia": round(adherencia, 3),
            "gravedad": round(gravedad, 3),
            "score": round(score, 2)
        })

    df_farmacos = pd.DataFrame(filas_farmacos)
    df_farmacos_sorted = df_farmacos.sort_values("score", ascending=False)
    recomendaciones.extend(df_farmacos_sorted.to_dict(orient="records"))

# Mostrar recomendaciones finales
recomendaciones_df = pd.DataFrame(recomendaciones)
print("\n💊 Recomendaciones por paciente:")
print(recomendaciones_df[["paciente_id", "farmaco", "eficacia", "adherencia", "gravedad", "score"]])
