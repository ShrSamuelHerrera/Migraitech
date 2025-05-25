# 📘 `multisalidaV2.py` — Entrenamiento de modelos para predicción multisalida (eficacia y adherencia)

Este script entrena dos modelos de Machine Learning para predecir:

1. **Eficacia** (clasificada como alta o no alta) → clasificación binaria  
2. **Adherencia** → regresión continua

---

## 🧠 ¿Qué hace este script?

- Carga un dataset de pacientes (`datos_pacientes_entrenamiento.xlsx`)
- Preprocesa variables clínicas y categóricas
- Clasifica la eficacia según el percentil 75 (definiendo un umbral personalizado)
- Entrena dos modelos:
  - 🎯 Un modelo de clasificación (`XGBoost`) para la eficacia
  - 📈 Un modelo de regresión (`RandomForest`) para la adherencia
- Evalúa y guarda ambos modelos junto con el encoder

---

## 🔍 Estructura paso a paso

### 1. Carga de datos

```python
df = pd.read_excel("datos_pacientes_entrenamiento.xlsx")
```

- Elimina columnas que solo contienen ceros
- Redondea variables como IMC, peso, eficacia, etc.

### 2. Clasificación de Eficacia

```python
p75 = df["Eficacia"].quantile(0.75)
df["Eficacia_clasificada"] = df["Eficacia"].apply(lambda x: "Alta" if x >= p75 else "No Alta")
```

- Se considera "Alta" eficacia si supera el percentil 75 de la muestra

### 3. Codificación y preparación

- Codifica `genero` numéricamente
- Aplica One-Hot Encoding a la variable `farmaco`
- Separa variables predictoras y variables objetivo

### 4. División de datos

- Se divide el dataset en entrenamiento y test para:
  - Eficacia (clasificación)
  - Adherencia (regresión)

### 5. Balanceo de clases

```python
X_train_bal, y_train_efi_clas_bal = SMOTE().fit_resample(...)
```

- Se usa SMOTE para equilibrar las clases de eficacia antes de entrenar

### 6. Entrenamiento de modelos

```python
model_efi_clas = XGBClassifier(...)
model_ade = RandomForestRegressor().fit(...)
```

- Se entrena un modelo para cada tarea

### 7. Guardado de modelos

```python
joblib.dump(..., "modelo_clas_eficacia.pkl")
joblib.dump(..., "modelo_rf_aderencia.pkl")
joblib.dump(..., "encoder_farmacos.pkl")
```

### 8. Evaluación

- Eficacia: usa matriz de confusión y métricas de clasificación
- Adherencia: usa R², MAE y RMSE

---

## ⚙️ Archivos generados

- `modelo_clas_eficacia.pkl` → Modelo de clasificación de eficacia
- `modelo_rf_aderencia.pkl` → Modelo de regresión para adherencia
- `encoder_farmacos.pkl` → Codificador de variable `farmaco`

---

## 🤖 ¿Por qué estos modelos?

### 1. Eficacia → `XGBoostClassifier`

🔎 **Objetivo**: Predecir si la eficacia del tratamiento será **alta** o **no alta**.

📌 **Motivo de elección**:
- `XGBoost` es altamente eficaz para clasificación tabular.
- Maneja variables clínicas mixtas y relaciones no lineales.
- Evita el sobreajuste gracias a su regularización.

✅ **Ventajas**:
- Alta precisión y rendimiento
- Buen manejo de datos clínicos estructurados
- Importancia de variables interpretable

---

### 2. Adherencia → `RandomForestRegressor`

🔎 **Objetivo**: Predecir la adherencia como un valor continuo.

📌 **Motivo de elección**:
- `Random Forest` es robusto frente a ruido y colinealidades.
- Muy adecuado para regresión clínica sin requerir normalización.

✅ **Ventajas**:
- Estabilidad y generalización
- Captura relaciones complejas sin ajustes manuales
- Fácil de usar y eficiente

---

## 🛠 Requisitos

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn imbalanced-learn joblib
```

---

## ▶️ Ejecución

```bash
python multisalidaV2.py
```

El script entrenará ambos modelos, los evaluará y mostrará los resultados gráficamente.
