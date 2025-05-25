# 📘 `entrenar_modelo_logistic.py` — Clasificación multinomial con regresión logística

Este script entrena un modelo de regresión logística multinomial para predecir el tratamiento farmacológico más probable, basado en el perfil clínico del paciente.

---

## 🧠 ¿Qué hace este script?

- Carga y preprocesa un dataset clínico (`datos_pacientes_entrenamiento.xlsx`)
- Estandariza variables numéricas
- Codifica variables categóricas (`genero`, `farmaco`)
- Equilibra la clase "Amitriptilina" (reducción de sobremuestreo)
- Entrena un modelo de clasificación con `LogisticRegression`
- Evalúa el modelo con métricas y matriz de confusión
- Guarda el modelo, encoder y scaler para uso futuro

---

## 🔍 Flujo de trabajo

### 1. Carga y normalización

- Estandariza `Eficacia` a rango [0, 1]
- Codifica `genero` como 0 (F) y 1 (M)
- Codifica `farmaco` con `LabelEncoder`

### 2. Balanceo de clases

- La clase "Amitriptilina" se reduce a 150 ejemplos aleatorios para evitar sesgo

### 3. Variables utilizadas

Incluye:
- edad, IMC_categorica, genero
- Eficacia, Adherencia, Gravedad Total
- Todas las columnas que empiezan con `Comorbilidad_` o `EA_`

### 4. Escalado y división

- Se escalan todas las variables predictoras con `StandardScaler`
- Se hace un `train_test_split` 80/20

### 5. Entrenamiento del modelo

```python
LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000, class_weight='balanced')
```

- Se entrena en modo multiclase con clases balanceadas

### 6. Guardado de artefactos

- `modelo_logistic_reg.pkl` → modelo entrenado
- `label_encoder_farmacos.pkl` → codificador de fármacos
- `scaler.pkl` → escalador de variables

### 7. Evaluación

- Accuracy
- F1 Score
- Reporte por clase
- Matriz de confusión gráfica (`seaborn`)

---

## 🤖 ¿Por qué regresión logística?

- Método interpretable y rápido para clasificación multiclase
- Funciona bien con datos estructurados
- Permite análisis probabilístico por clase
- Útil como modelo base o comparativo frente a árboles y redes

---

## 📂 Archivos generados

- `modelo_logistic_reg.pkl`
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
python entrenar_modelo_logistic.py
```

Generará el modelo entrenado y evaluará su rendimiento en consola y gráfico.

