# Protocolo experimental

## Decisiones ya tomadas

### 1. Duplicados que cruzan particiones
**Decisión:** cuando una misma imagen con la misma etiqueta aparece en dos particiones, se conserva la copia de test (o la de validation si el cruce es con train) y se elimina la otra.
**Justificación:** el conjunto de prueba funciona como el examen final: debe quedar fijo para que todos los experimentos se evalúen con las mismas imágenes y para mantener las particiones oficiales de TFDS. Por eso se quitan las copias de train, que es con lo que el modelo aprende. Así se cumple el requisito de no repartir capturas repetidas entre entrenamiento y prueba.
**Resultado:** 6 imágenes excluidas (5 de train, 1 de validation).

### 2. Imágenes con etiquetas contradictorias
**Decisión:** se excluyen todas las imágenes de los grupos de duplicados con etiquetas distintas, en todas las particiones, incluido test. Esta regla tiene prioridad sobre la decisión 1.
**Justificación:** en la mayoría de esos grupos la misma foto aparece una vez como cbb, otra como cbsd y otra como cgm; no hay etiqueta mayoritaria y no hay forma de saber cuál es la correcta. En train, el modelo aprendería información contradictoria; en test, se evaluaría contra una respuesta que puede estar mal. Posibles causas (no comprobadas, se manejan como hipótesis): error de etiquetado, síntomas parecidos entre enfermedades o una hoja con varias enfermedades.
**Resultado:** 56 imágenes excluidas (35 de train, 10 de validation, 11 de test).
**Limitación:** las exclusiones se concentran en clases minoritarias (cbb, cbsd y cgm), y el test final tiene 11 imágenes menos que el oficial (0,6 %).

### 3. Tamaño de imagen y redimensionamiento
**Decisión:** redimensionar el lado corto a 256 píxeles y recortar el centro a 224×224, igual en train, validation y test y en todos los experimentos.
**Justificación:** es el preprocesamiento oficial de los pesos preentrenados de MobileNetV2 y ResNet18 en torchvision (verificado: resize 256, recorte 224, interpolación bilineal), así que las imágenes llegan al modelo en la forma para la que fue entrenado. No deforma las lesiones, y como todas las imágenes terminan cuadradas sin deformación ni franjas, la proporción de la foto (que varía por clase) no puede servir como atajo.
**Limitación:** se pierden los bordes; en fotos de 888×500 se descarta casi la mitad del ancho. Se revisará en el análisis de errores.

### 4. Detección de duplicados
**Decisión:** MD5 para copias exactas y pHash con umbral 0 para casi duplicados.
**Justificación:** con umbral 0 se detectaron 4 pares casi idénticos adicionales, revisados visualmente. Un umbral mayor podría marcar como duplicadas fotos distintas que solo se parecen.
**Limitación:** pueden quedar fotos de la misma hoja tomadas en momentos distintos que este método no detecta.

### Particiones finales
train 5616 · validation 1878 · test 1874 (archivo: results/particiones_finales.csv; exclusiones con su motivo en results/imagenes_excluidas.csv).

### Normalización
Media [0.485, 0.456, 0.406] y desviación estándar [0.229, 0.224, 0.225] de ImageNet, la que corresponde a ambos backbones.

## Decisiones pendientes (reunión del equipo)
## Configuración de entrenamiento y evaluación

### 5. Semilla
**Decisión:** semilla 42 en todos los experimentos.
**Justificación:** el valor de la semilla es arbitrario; lo importante es usar la misma en todos los experimentos para que el orden de los datos y los pesos iniciales sean iguales y los resultados se puedan repetir.
**Limitación:** en GPU el determinismo no es 100 % garantizado, aunque se fije la semilla.

### 6. Métrica principal
**Decisión:** macro-F1. También se reportan accuracy, y recall y F1 por clase.
**Justificación:** las clases están muy desbalanceadas (cmd es el 47,2 % del test y healthy el 5,6 %). La accuracy puede verse bien aunque el modelo falle en las clases pequeñas: un modelo que siempre responda "cmd" tendría 47,2 % de accuracy pero solo 12,8 % de macro-F1. Macro-F1 promedia el F1 de las cinco clases con el mismo peso, así que muestra si el modelo realmente reconoce cada enfermedad y no solo la más frecuente.

### 7. Presupuesto de entrenamiento
**Decisión:** 10 épocas fijas para E1 y E3. E2 parte del modelo final de E1 y entrena 2 épocas adicionales.
**Justificación:** al entrenar solo una cabeza lineal sobre un backbone congelado hay pocos parámetros que ajustar, y el modelo suele estabilizarse en pocas épocas. Como se guarda la época con mejor macro-F1 en validación (decisión 11), tener algunas épocas de margen no perjudica el resultado si el modelo se sobreajusta; solo aumenta el tiempo. En E2 se usan menos épocas porque el modelo ya parte entrenado y la tasa de aprendizaje es menor; solo se busca ajustar. Con el mismo número de épocas, E1 y E3 tienen el mismo presupuesto y la comparación es justa. Preferimos épocas fijas a early stopping para no agregar otro parámetro. El costo de E2 se reporta como el de E1 más sus 2 épocas adicionales.
**Limitación:** con 2 épocas, la curva de E2 tiene muy pocos puntos, así que será difícil ver si el ajuste fino seguiría mejorando o empezaría a sobreajustar.

### 8. Optimizador y tasa de aprendizaje
**Decisión:** Adam. Tasa de aprendizaje de 0,001 (valor por defecto de Adam) para la cabeza en E1 y E3, y de 0,0001 para E2.
**Justificación:** Adam ajusta automáticamente el tamaño del paso para cada peso, así que funciona bien sin tener que afinar mucho la tasa de aprendizaje, lo que es útil con el tiempo que tenemos. Se usa la misma configuración en E1 y E3 para que la única diferencia sea la arquitectura. En E2 la tasa es diez veces menor, como pide el taller, para ajustar los pesos preentrenados sin borrar lo que aprendieron en ImageNet.

### 9. Batch size
**Decisión:** 32 en E1, E2 y E3.
**Justificación:** es un tamaño común, cabe sin problema en la memoria de la GPU T4 y hace unas 176 actualizaciones de los pesos por época. Se usa el mismo valor en todos los experimentos para que sean comparables.

### 10. Manejo del desbalance
**Decisión:** no se aplica ninguna técnica de balanceo (ni pérdida ponderada ni muestreo); el desbalance se analiza y se reporta.
**Justificación:** queremos comparar las arquitecturas y los regímenes de entrenamiento en las mismas condiciones, sin que una técnica adicional influya en los resultados. Como la métrica principal es macro-F1 y se reportan recall y F1 por clase, el efecto del desbalance va a quedar visible y se podrá analizar.
**Limitación:** es probable que las clases minoritarias (healthy y cbb) tengan menor recall.

### 11. Criterio de selección del modelo
**Decisión:** en cada experimento se guarda la época con mejor macro-F1 en validación, y ese es el modelo que se evalúa en test.
**Justificación:** si el modelo empieza a sobreajustarse en las últimas épocas, la última época no sería la mejor. Se usa validación para elegir y nunca test, para que test siga siendo una evaluación independiente. Se usa macro-F1 porque es la métrica principal.

### 12. Baseline E0
**Decisión:** regresión logística sobre histogramas de color. A cada imagen (después del mismo resize 256 y recorte 224 de los demás experimentos) se le calcula un histograma de 32 intervalos por canal (rojo, verde y azul), lo que da 96 valores por imagen.
**Justificación:** es un modelo simple, rápido y fácil de explicar, que responde la pregunta del taller: qué se consigue sin una arquitectura sofisticada. El histograma solo resume qué colores hay en la imagen y en qué cantidad, sin tener en cuenta formas ni posiciones. Por eso, además de servir como punto de comparación para medir cuánto aporta el aprendizaje por transferencia, nos ayuda a saber cuánto se puede acertar solo con el color. Si E0 obtiene un resultado alto, sería un indicio de que el color (incluido el del fondo o la iluminación) puede estar funcionando como atajo, algo que el taller pide analizar.
**Limitación:** al ignorar la forma y la posición de las lesiones, no puede distinguir enfermedades que producen colores parecidos.

### 13. Cabeza clasificadora (E1 y E3)
**Decisión:** una sola capa lineal, de las características del backbone a las 5 clases.
**Justificación:** es la cabeza más simple; al usar la misma en E1 y E3, las diferencias de resultado se pueden atribuir al backbone y no a la cabeza.

### 14. Estabilidad
**Decisión:** una sola ejecución por experimento, con la semilla 42.
**Justificación:** por el tiempo disponible y los recursos de Colab gratuito, no es posible repetir cada experimento varias veces.
**Limitación:** no podemos medir cuánto varían los resultados entre ejecuciones; si la diferencia entre dos modelos es pequeña, no podremos afirmar que sea real. Las curvas de entrenamiento y validación se usarán como indicio de estabilidad.

### 15. Experimento adicional
**Decisión:** se hará sobre E3. El tipo (augmentation moderado en el entrenamiento de E3, o perturbación de brillo sobre una copia del test evaluada con E3) se decidirá después de ejecutar E1 y E3, y su objetivo se declarará aquí antes de ejecutarlo.
**Justificación:** el taller permite cualquiera de las dos opciones. Decidir después de tener E3 nos permite elegir la que aporte más al análisis sin retrasar los experimentos principales.

### Alcance de los hiperparámetros
Las decisiones 7, 8, 9 y 13 aplican a E1, E2 y E3. E0 (regresión logística) se entrena con scikit-learn, por lo que no usa épocas, Adam ni batch size.
