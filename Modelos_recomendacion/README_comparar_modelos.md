# 📘 `comparar_modelos.py` — Comparación de modelos clasificadores multiclase

Este script evalúa y compara el rendimiento de dos modelos de clasificación multiclase previamente entrenados:

- 🔹 Regresión logística multinomial
- 🔸 Random Forest

---

## 🧠 ¿Qué hace este script?

1. Carga los datos clínicos desde Excel (`datos_pacientes_entrenamiento.xlsx`)
2. Preprocesa los datos (normalización, codificación, selección de variables)
3. Aplica un `LabelEncoder` ya entrenado para codificar la variable `farmaco`
4. Divide los datos en entrenamiento y test (20%)
5. Carga los modelos entrenados:
   - `modelo_logistic_reg.pkl`
   - `modelo_random_forest.pkl`
6. Escala los datos solo para el modelo de regresión logística
7. Evalúa ambos modelos:
   - Accuracy
   - F1 Score ponderado
   - Recall promedio
   - Matriz de confusión
8. Muestra gráficamente las matrices de confusión

---

## 🔍 Características utilizadas

- edad
- IMC_categorica
- genero (0/1)
- Eficacia (normalizada)
- Adherencia
- Gravedad Total
- Todas las columnas que comienzan con `Comorbilidad_` o `EA_`

---

## 🧪 Evaluación

Cada modelo se evalúa con las siguientes métricas:

- Accuracy global
- F1 Score promedio ponderado
- Recall promedio
- Reporte de clasificación por clase
- Matriz de confusión (heatmap con seaborn)

---

## 🤖 ¿Qué modelos compara?

1. **Regresión Logística**
   - Entrenado con `StandardScaler`
   - Evalúa probabilidades para cada clase
2. **Random Forest**
   - Modelo no paramétrico
   - No requiere escalado
   - Robusto frente a ruido y datos no lineales

---

## 📂 Archivos requeridos

- `datos_pacientes_entrenamiento.xlsx`
- `modelo_logistic_reg.pkl`
- `modelo_random_forest.pkl`
- `label_encoder_farmacos.pkl`
- `scaler.pkl`

---

## 🛠 Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

---

## ▶️ Ejecución

```bash
python comparar_modelos.py
```

Generará métricas y gráficos que te ayudarán a elegir el modelo más adecuado para tus predicciones.

