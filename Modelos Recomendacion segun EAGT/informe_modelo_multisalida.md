
# Justificación del cambio de enfoque: del modelo clásico al sistema multisalida

Aunque el modelo original basado en Random Forest incluía variables como la eficacia y la adherencia dentro del conjunto de entrenamiento, estas no eran el objetivo del modelo, sino simplemente **variables predictoras**. El modelo estaba entrenado para **predecir directamente el fármaco que fue prescrito históricamente**, sin evaluar explícitamente si esa decisión fue clínicamente adecuada.

## ❌ Limitaciones del enfoque tradicional

1. **No separa calidad y cantidad**  
   Si un fármaco fue muy recetado, el modelo tenderá a sugerirlo, incluso si tuvo baja eficacia o mala adherencia.

2. **Eficacia y adherencia no son criterios clínicos explícitos**  
   Aunque estén en los datos, el modelo no intenta optimizarlos; solo los usa como pistas para reproducir decisiones pasadas.

3. **Falta de interpretabilidad clínica**  
   No permite saber si el fármaco recomendado lo es por su efectividad real, por su popularidad histórica o por casualidad estadística.

4. **No permite personalización ni reglas clínicas**  
   El modelo es una caja negra que generaliza, pero no permite aplicar restricciones específicas por paciente, como contraindicaciones clínicas.

## 🍽️ Metáfora: ¿Elegimos por costumbre o por calidad?

> Imagina que tienes un historial de restaurantes que has visitado. Puedes entrenar un modelo que diga:
>
> - *“Cuando era la 1 p.m. y estabas en el centro, fuiste al restaurante X”* → eso es el **modelo Random Forest**: imita decisiones anteriores.
>
> Pero otro modelo puede decir:
>
> - *“De todos los restaurantes disponibles, este tiene mejor puntuación en sabor y servicio, para tus preferencias”* → eso es el **modelo multisalida**: evalúa cada opción según criterios relevantes para ti.

## ✅ Enfoque Multisalida: predicción clínicamente informada

El nuevo enfoque responde a dos preguntas clínicas fundamentales:

1. **¿Será este fármaco eficaz para este paciente?**
2. **¿Es probable que este paciente siga correctamente el tratamiento?**

Para ello, se construyen dos modelos independientes:

### 🔹 Modelo 1: Clasificación de la eficacia (`Eficacia_clasificada`)

- **Variable objetivo**: una versión binarizada de `Eficacia`, usando el percentil 75 como umbral. Se clasifica como:
  - `"Alta"` si eficacia ≥ P75.
  - `"No Alta"` en caso contrario.

#### 📌 ¿Por qué se usa un umbral basado en percentiles?

La distribución de la variable `Eficacia` en los datos originales es asimétrica: la mayoría de los valores están concentrados en rangos bajos y solo unos pocos pacientes presentan respuestas significativamente más altas. Esto refleja la naturaleza preventiva del tratamiento, que raramente produce efectos espectaculares.

Usar un umbral fijo para definir “eficacia alta” sería arbitrario e inadecuado. En cambio, el percentil 75 permite:

- **Adaptarse a la distribución real de los datos.**
- **Detectar los mejores resultados relativos**, aunque los valores absolutos sean bajos.
- **Equilibrar mejor las clases** al entrenar el modelo.

Este enfoque ofrece una forma más robusta y coherente de identificar casos de alta eficacia en función del contexto clínico observado.

- **Variables predictoras**: Edad, género, IMC, peso, comorbilidades, efectos adversos, gravedad total y fármaco (one-hot encoded).
- **Utilidad clínica**: Permite predecir si el tratamiento tendrá un impacto destacado, según el perfil clínico del paciente.

### 🔹 Modelo 2: Regresión de la adherencia (`Adherencia`)

- **Variable objetivo**: `Adherencia`, medida continua de seguimiento al tratamiento.

- **Variables predictoras**: mismas que en el modelo anterior, incluyendo carga clínica y características personales.

- **Utilidad clínica**: Estima, en base a datos objetivos, qué tan probable es que el paciente cumpla con el tratamiento prescrito.

### 🔄 Combinación de predicciones

Para cada paciente, se evalúan todos los fármacos posibles y se calcula:

- La probabilidad de eficacia alta.
- El nivel esperado de adherencia.

Ambas se combinan en un **score final ponderado**, por ejemplo:

```python
score = (eficacia_proba * 0.6) + (adherencia * 0.4)
```

Esto permite **comparar opciones y recomendar la mejor combinando ambos criterios**.

## 📈 Ventajas del enfoque multisalida

- Genera un **ranking personalizado de tratamientos** para cada paciente.
- Es **modular y explicable**: se puede justificar cada recomendación con evidencia.
- Permite aplicar **reglas clínicas personalizadas** (por ejemplo, restricciones por edad, embarazo, comorbilidades).
- Se enfoca en **maximizar el resultado clínico**, no en repetir el historial.

## 🧠 Conclusión

> Mientras que el modelo Random Forest reproduce lo que históricamente se hizo, el sistema multisalida **evalúa lo que clínicamente se debería hacer**, en base a eficacia esperada y adherencia estimada. Esto convierte el sistema en una herramienta de recomendación médica más segura, más transparente y centrada en el paciente.
