# Hipótesis previas a los experimentos
Redactadas por el equipo el 2026-11-16, antes de ejecutar cualquier experimento.
Métrica principal para comparar: macro-F1 en test (protocolo, decisión 6).

## E0. Baseline (regresión logística sobre histogramas de color)
**Hipótesis:** Se espera que un modelo sencillo pueda identificar algunas características de las hojas de yuca y clasificarlas en las cinco categorías, aunque puede presentar dificultades al diferenciar enfermedades con síntomas similares.

## E1. Extracción de características (MobileNetV2 congelada) frente a E0
**Hipótesis:** Se espera que utilizar un modelo preentrenado permita mejorar la clasificación de las enfermedades, aprovechando características que el modelo ya ha aprendido, sin necesidad de entrenar todas sus capas.

## E2. Fine-tuning parcial (último bloque de MobileNetV2) frente a E1
**Hipótesis:** Se espera que descongelar el último bloque del modelo permita adaptarlo mejor a las imágenes de hojas de yuca y mejorar los resultados obtenidos en el experimento anterior.

## E3. Segunda arquitectura (ResNet18 congelada) frente a E1
**Hipótesis:** Se espera que la segunda arquitectura preentrenada presente diferencias en el reconocimiento de las enfermedades frente a la primera, permitiendo identificar cuál ofrece mejores resultados con un costo de entrenamiento razonable.

## E4. Experimento adicional: augmentation moderado en el entrenamiento de E3
**Hipótesis:** Se espera que aplicar pequeñas transformaciones a las imágenes de entrenamiento ayude al modelo a reconocer las enfermedades en diferentes condiciones, mejorando su capacidad para clasificar imágenes que no ha visto anteriormente.

