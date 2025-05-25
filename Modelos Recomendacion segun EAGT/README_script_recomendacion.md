# 📘 `script_recomendacion.py` — Recomendación personalizada de tratamientos farmacológicos

Este script utiliza modelos previamente entrenados para recomendar el fármaco más adecuado a nuevos pacientes, en función de su perfil clínico, eficacia esperada y adherencia pronosticada.

---

## 🧠 ¿Qué hace este script?

1. Carga modelos previamente entrenados:
   - Clasificación de eficacia (`modelo_clas_eficacia.pkl`)
   - Regresión de adherencia (`modelo_rf_aderencia.pkl`)
2. Carga un archivo Excel con nuevos pacientes (`nuevos_pacientes.xlsx`)
3. Aplica filtros de exclusión según el tratamiento actual y restricciones clínicas.
4. Evalúa la eficacia y adherencia esperadas para cada fármaco candidato.
5. Devuelve una tabla con recomendaciones ordenadas por score combinado.

---

## 🔍 Flujo de trabajo

### 1. Carga y preprocesado

- Redondeo de valores de peso e IMC.
- Mapeo de género a valores numéricos.
- Cálculo de categoría de IMC (`IMC_categoria`)
- Cálculo de gravedad total si no está presente.

### 2. Restricciones clínicas

Definidas por fármaco, se descartan candidatos si el paciente tiene condiciones incompatibles como:

- Trastorno bipolar, embarazo, asma, EPOC, etc.
- Mujeres jóvenes (18–38 años) para Topiramato
- IMC bajo o normal también restringe Topiramato

### 3. Exclusión del tratamiento actual

Se evita recomendar el mismo fármaco que ya está tomando el paciente.

### 4. Predicciones

- Se codifica la variable `farmaco` con el encoder entrenado.
- Se predicen:
  - Probabilidad de eficacia alta (clasificación)
  - Valor de adherencia (regresión)
- Se calcula un score compuesto:
  - `score = 60% eficacia + 40% adherencia`

### 5. Salida

Una tabla final muestra por paciente:

- `paciente_id`, `farmaco`, `eficacia`, `adherencia`, `score`

---

## 🤖 Modelos utilizados

- **Eficacia** → `XGBoostClassifier` (ver script de entrenamiento)
- **Adherencia** → `RandomForestRegressor`
- **Codificación de fármaco** → `OneHotEncoder`

---

## 📂 Archivos necesarios

- `modelo_clas_eficacia.pkl`
- `modelo_rf_aderencia.pkl`
- `encoder_farmacos.pkl`
- `nuevos_pacientes.xlsx`

---

## ▶️ Ejecución

```bash
python script_recomendacion.py
```

El sistema devolverá recomendaciones personalizadas por paciente.

---

## 🛠 Requisitos

```bash
pip install pandas numpy scikit-learn xgboost joblib openpyxl
```

---

## 💡 Notas

- Las restricciones se definen en la función `es_farmaco_permitido()`.
- Puedes añadir más reglas sin afectar al funcionamiento del sistema.
