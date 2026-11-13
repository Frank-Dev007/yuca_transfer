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
- **Semilla:**
- **Métrica principal:**
- **Presupuesto de entrenamiento** (épocas fijas o early stopping; épocas adicionales de E2):
- **Optimizador y tasa de aprendizaje** (cabeza en E1/E3; tasa menor en E2):
- **Batch size:**
- **Manejo del desbalance:**
- **Criterio de selección del modelo** (mejor época según validación o última época):
- **Tipo de baseline E0:**
- **Número de semillas para estabilidad:**
- **Experimento adicional** (augmentation en train o brillo en una copia de test):
