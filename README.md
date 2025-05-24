
# Comparación de Modelos para Recomendación de Fármacos en Migraña

Este proyecto compara dos enfoques de clasificación para predecir el fármaco más adecuado para pacientes con migraña, basándose en variables clínicas como edad, eficacia, adherencia, comorbilidades y efectos adversos.

Se han entrenado y evaluado dos modelos sobre el mismo conjunto de datos:

- **Regresión Logística Multinomial (escalada con StandardScaler)**
- **Random Forest Classifier con `class_weight='balanced'`**

Ambos modelos fueron validados usando el mismo conjunto de test con 406 pacientes.

---

## 📊 Resultados Comparativos

| Modelo               | Accuracy | F1 Score | Recall Promedio |
|----------------------|----------|----------|------------------|
| Regresión Logística  | 80.5%    | 0.8224   | 0.8054           |
| Random Forest        | 86.7%    | 0.8607   | 0.8670           |

---

## 🧠 Análisis

- **Random Forest** mostró un mejor rendimiento general, con mayor precisión y recall en casi todas las clases.
- Ambos modelos identificaron perfectamente los casos de **Propranolol**, con 100% de acierto.
- **Regresión Logística** mostró un mejor recall para algunos fármacos minoritarios como **Topiramato**, pero con menor precisión.
- **Random Forest** logró un mayor equilibrio global entre precisión y cobertura de clases.

---

## ✅ Conclusión

Se recomienda usar el modelo de **Random Forest** como predictor principal, dada su robustez y rendimiento global superior.

La regresión logística puede ser útil como modelo complementario por su interpretabilidad y facilidad de análisis clínico.


---

## 🧾 Conclusión Final

El modelo Random Forest ha sido validado frente a una regresión logística multinomial, entrenada y evaluada bajo las mismas condiciones.  
Los resultados muestran que Random Forest obtiene un rendimiento superior en precisión, F1-score y capacidad de detección de casos reales (recall).

Esta validación justifica su elección como modelo principal para la recomendación de tratamiento farmacológico en pacientes con migraña.
