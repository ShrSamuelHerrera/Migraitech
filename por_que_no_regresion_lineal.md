
# ❌ ¿Por qué no usar Regresión Lineal para Clasificación?

Aunque su nombre pueda sugerir lo contrario, **la regresión lineal no es adecuada para tareas de clasificación** como la predicción de qué fármaco es el más adecuado para un paciente.

---

## 🧠 Limitaciones fundamentales de la regresión lineal en clasificación:

### 1. ❌ Predice valores continuos
La regresión lineal está diseñada para predecir un número real (ej. 2.73), **no una clase discreta**.  
En cambio, la clasificación requiere decidir entre **categorías** (ej. “Topiramato”, “Erenumab”).

### 2. ❌ Salidas no acotadas
Las predicciones de una regresión lineal pueden ser:
- Negativas
- Mayores a 1
- No interpretables como probabilidades

### 3. ❌ No permite clasificación multiclase directa
La regresión lineal no puede manejar múltiples clases como salida natural.  
Intentar usarla para eso implica forzar una codificación artificial que **distorsiona los resultados**.

---

## ✅ ¿Qué usar en su lugar?

### ✔️ Regresión Logística
- Usa la **función logística (sigmoide o softmax)** para convertir valores en **probabilidades**
- Es el modelo estadístico estándar para clasificación binaria y multiclase
- Devuelve probabilidades para cada clase, lo cual es interpretable y útil clínicamente

---

## 🎯 En resumen:

| Aspecto                       | Regresión Lineal ❌ | Regresión Logística ✅ |
|-------------------------------|---------------------|------------------------|
| Tipo de salida                | Numérica continua   | Probabilidad / Clase   |
| Acotación de salida           | No (puede ser <0 o >1) | Sí (entre 0 y 1)   |
| Adecuado para clasificación   | No                  | Sí                     |
| Compatible con métricas como F1 | No               | Sí                     |

---

📌 Por todo esto, en este proyecto se usó **regresión logística multinomial** como modelo de clasificación base, y **Random Forest** como modelo final, descartando explícitamente el uso de regresión lineal.
