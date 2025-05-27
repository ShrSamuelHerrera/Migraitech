
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar modelos entrenados y codificador one-hot para farmacos
model_efi_clas = joblib.load("modelo_clas_eficacia.pkl")
model_ade = joblib.load("modelo_rf_aderencia.pkl")
encoder = joblib.load("encoder_farmacos.pkl")

# Cargar los datos de nuevos pacientes que aún no tienen fármaco asignado
nuevos_pacientes = pd.read_excel("nuevos_pacientes.xlsx")

# Recalcular la categoría del IMC si la columna IMC está presente
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

# Redondear valores para estandarizar
nuevos_pacientes["peso"] = nuevos_pacientes["peso"].round(1)
nuevos_pacientes["IMC"] = nuevos_pacientes["IMC"].round(1)

# Codificar el género como numérico: 0 para Femenino, 1 para Masculino
if nuevos_pacientes["genero"].dtype == object:
    nuevos_pacientes["genero"] = nuevos_pacientes["genero"].map({"Femenino": 0, "Masculino": 1})

# Calcular "Gravedad Total" si no existe, sumando comorbilidades y efectos adversos
if "Gravedad Total" not in nuevos_pacientes.columns:
    columnas_gravedad = [col for col in nuevos_pacientes.columns if "ef_adverso" in col or "cormo" in col]
    nuevos_pacientes["Gravedad Total"] = nuevos_pacientes[columnas_gravedad].sum(axis=1)

# Función para verificar si un fármaco está permitido clínicamente para un paciente
def es_farmaco_permitido(paciente, farmaco):
    restricciones = {
        "Amitriptilina": [
            "cormo_Trastorno bipolar", "cormo_Manía",
            "cormo_Enfermedades cardiovasculares", "cormo_Hepatopatía grave"
        ],
        "Propranolol": [
            "cormo_Enfermedades cardiovasculares", "cormo_Acidosis metabólica",
            "cormo_Asma", "cormo_EPOC", "cormo_Síndrome Raynaud",
            "cormo_Angina hipoglicémica"
        ],
        "Erenumab": ["cormo_Embarazo", "cormo_Lactancia"],
        "Fremanezumab": ["cormo_Embarazo", "cormo_Lactancia"],
        "Galcanezumab": ["cormo_Embarazo", "cormo_Lactancia"],
        "Topiramato": ["cormo_Embarazo", "cormo_Lactancia"]
    }

    if farmaco in restricciones:
        for col in restricciones[farmaco]:
            if paciente.get(col, 0) == 1:
                return False

    # Restricciones específicas para Topiramato
    if farmaco == "Topiramato":
        edad = paciente.get("edad", None)
        genero = paciente.get("genero", None)
        imc_cat = paciente.get("IMC_categoria", None)

        if genero == 0 and edad is not None and 18 <= edad <= 38:
            return False

        if imc_cat in [1, 2]:
            return False

    return True

# Lista de fármacos posibles desde el encoder (usados en entrenamiento)
farmacos = encoder.categories_[0]

# Generar recomendaciones de fármacos para cada paciente
recomendaciones = []
for idx, paciente in nuevos_pacientes.iterrows():
    farmaco_actual = paciente.get("farmaco", None)
    filas_farmacos = []
    for farmaco in farmacos:
        if farmaco_actual == farmaco:
            continue  # No recomendar el mismo fármaco
        if not es_farmaco_permitido(paciente, farmaco):
            continue  # Omitir si hay contraindicaciones

        paciente_mod = paciente.copy()
        paciente_mod["farmaco"] = farmaco
        paciente_df = pd.DataFrame([paciente_mod])

        # Aplicar one-hot encoding al nuevo fármaco
        X_encoded = encoder.transform(paciente_df[["farmaco"]])
        X_otros = paciente_df.drop(columns=["farmaco"]).reset_index(drop=True)
        X_final = pd.concat([X_otros, pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["farmaco"]))], axis=1)

        # Asegurar que todas las columnas requeridas por el modelo estén presentes
        columnas_modelo = model_efi_clas.feature_names_in_
        faltantes = [col for col in columnas_modelo if col not in X_final.columns]
        if faltantes:
            nuevos = pd.DataFrame(0, index=np.arange(len(X_final)), columns=faltantes)
            X_final = pd.concat([X_final, nuevos], axis=1)

        X_final = X_final[columnas_modelo]
        X_final.columns = [col.replace(" ", "_") for col in X_final.columns]

        # Obtener predicción de eficacia y adherencia
        eficacia_proba = model_efi_clas.predict_proba(X_final)[0][1]
        adherencia = model_ade.predict(X_final)[0]

        # Calcular un score compuesto ponderando eficacia (60%) y adherencia (40%)
        raw_score = eficacia_proba * 0.6 + adherencia * 0.4
        score = max(0, min((raw_score / 1.0) * 100, 100))

        filas_farmacos.append({
            "paciente_id": paciente["id"],
            "farmaco": farmaco,
            "eficacia": round(eficacia_proba, 3),
            "adherencia": round(adherencia, 3),
            "score": round(score, 2)
        })

    # Ordenar por mayor score y añadir a la lista de recomendaciones finales
    df_farmacos = pd.DataFrame(filas_farmacos)
    df_farmacos_sorted = df_farmacos.sort_values("score", ascending=False)
    recomendaciones.extend(df_farmacos_sorted.to_dict(orient="records"))

# Mostrar resultados finales
recomendaciones_df = pd.DataFrame(recomendaciones)
print("\n💊 Recomendaciones por paciente:")
print(recomendaciones_df[["paciente_id", "farmaco", "eficacia", "adherencia", "score"]])
