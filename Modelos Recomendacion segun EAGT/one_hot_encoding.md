
# ¿Qué es One-Hot Encoding?

**One-Hot Encoding** es una técnica de preprocesamiento utilizada para convertir variables categóricas en variables numéricas, lo cual es necesario para la mayoría de algoritmos de machine learning que no pueden manejar datos no numéricos directamente.

---

## ¿Cómo funciona?

Dado un conjunto de valores categóricos, One-Hot Encoding crea una nueva columna para cada categoría y asigna un valor de 1 o 0 para indicar la presencia o ausencia de esa categoría en una observación.

### Ejemplo

Supongamos una columna con el nombre del fármaco:

| farmaco |
|---------|
| A       |
| B       |
| C       |

Después de aplicar One-Hot Encoding, obtenemos:

| farmaco_A | farmaco_B | farmaco_C |
|-----------|-----------|-----------|
| 1         | 0         | 0         |
| 0         | 1         | 0         |
| 0         | 0         | 1         |

---

## ¿Por qué se utiliza?

- **Los modelos de machine learning no entienden texto**: necesitan números para poder calcular distancias, probabilidades, etc.
- **Evita relaciones falsas**: a diferencia del Label Encoding, no impone un orden o jerarquía entre categorías.
- **Compatible con muchos algoritmos**: como regresión, redes neuronales, árboles, etc.

---

## ¿Cómo se aplica en Python?

En el script proporcionado se usa así:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[["farmaco"]])
```

- `sparse_output=False`: devuelve un array denso en lugar de una matriz dispersa.
- `handle_unknown="ignore"`: evita errores si aparecen categorías nuevas en datos de prueba.

---

## Ventajas

✅ Intuitivo y fácil de implementar  
✅ Evita supuestos de orden entre categorías  
✅ Bien soportado por librerías de ML

## Desventajas

❌ Puede generar muchas columnas si hay muchas categorías  
❌ No es ideal para variables categóricas de alta cardinalidad

---

## Conclusión

One-Hot Encoding es una técnica clave para convertir datos categóricos en un formato adecuado para modelos de machine learning, sin introducir sesgos de orden o jerarquía.
