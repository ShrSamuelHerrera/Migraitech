# 📘 `random_forest_predictor.py` — Clasificador multiclase con Random Forest

Este script entrena un modelo `RandomForestClassifier` para predecir cuál es el fármaco más probable para un paciente, en función de su perfil clínico.

---

## 🧠 ¿Qué hace este script?

- Carga y preprocesa un dataset clínico (`datos_pacientes_entrenamiento.xlsx`)
- Codifica variables categóricas
- Selecciona características clínicas relevantes
- Entrena un clasificador multiclase con Random Forest
- Evalúa el modelo con métricas estándar y matriz de confusión
- Muestra la importancia de las variables predictoras
- Guarda el modelo y el codificador para uso posterior

---

## 🔍 Flujo de trabajo

### 1. Preprocesamiento

- Normaliza `Eficacia` al rango [0, 1]
- Codifica `genero` y `farmaco` con `LabelEncoder`
- Extrae características:
  - edad, IMC_categorica, genero
  - Eficacia, Adherencia, Gravedad Total
  - Todas las columnas que empiezan por `Comorbilidad_` o `EA_`

### 2. División de datos

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

20% de los datos se usa para evaluación.

### 3. Entrenamiento

```python
RandomForestClassifier(n_estimators=100, class_weight='balanced')
```

- Equilibrio de clases automático
- Entrenamiento en paralelo (`n_jobs=-1`)

### 4. Evaluación

- Accuracy
- Recall y F1 Score (promedio ponderado)
- Reporte de clasificación por clase
- Matriz de confusión con `seaborn`

### 5. Interpretabilidad

- Se imprimen las 10 variables más importantes para la predicción

### 6. Guardado

- `modelo_random_forest.pkl` → modelo entrenado
- `label_encoder_farmacos_rf.pkl` → codificador de `farmaco`

---

## 🤖 ¿Por qué Random Forest?

- Maneja bien variables clínicas heterogéneas
- Captura relaciones no lineales sin requerir ajustes
- Alta robustez frente a sobreajuste
- Funciona bien en clasificación multiclase con datasets estructurados

---

## 🛠 Requisitos

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib openpyxl
```

---

## ▶️ Ejecución

```bash
python random_forest_predictor.py
```

Se mostrará la evaluación y se guardarán los modelos entrenados para futura inferencia.

