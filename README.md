# Comparación de modelos preentrenados para reconocer enfermedades de la yuca

Taller de **Aprendizaje Computacional** · Ingeniería de Sistemas · Universidad de Córdoba (Colombia)
Equipo: Frank Manuel García Pernett, Tomás David González Lopez y Juan Camilo Díaz Blandon

Experimento controlado que compara un modelo base y redes convolucionales preentrenadas (aprendizaje por transferencia) para clasificar fotos de hojas de yuca en 5 clases: **cbb** (plaga bacteriana), **cbsd** (raya marrón), **cgm** (ácaro verde), **cmd** (mosaico) y **healthy** (sana). Todos los experimentos usan las mismas particiones, preprocesamiento, semilla y métricas.

## Resultados (test)

| Exp. | Modelo | Accuracy | Macro-F1 | Entrenamiento (s) |
|---|---|---:|---:|---:|
| E0 | Regresión logística sobre histogramas de color | 0,5491 | 0,2997 | 2,5 |
| E1 | MobileNetV2 congelada + cabeza lineal | 0,7689 | 0,6944 | 524 |
| E2 | MobileNetV2 con el último bloque descongelado (desde E1) | 0,8042 | 0,7323 | 641 (E1 + E2) |
| E3 | ResNet18 congelada + cabeza lineal | 0,7727 | 0,6842 | 518 |
| E4 | E3 + augmentation moderado en train (experimento adicional) | 0,7668 | 0,6577 | 544 |

Tabla completa (tiempos de inferencia y parámetros) en `results/tabla_consolidada.csv`. Métrica principal: **macro-F1**.

## Datos
- Conjunto: **cassava (iCassava 2019)** de TensorFlow Datasets, versión 0.1.0. Ficha: https://www.tensorflow.org/datasets/catalog/cassava
- Descarga del zip (1,35 GB): https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip
- Las imágenes **no están en el repositorio**. Para ejecutar, subir el zip a Google Drive en `Mi unidad/yuca_datos/cassavaleafdata.zip`; `scripts/preparar_datos.py` lo copia y descomprime en cada sesión. Alternativa sin Drive: `scripts/descargar_datos.py`.
- Se excluyeron 62 imágenes repetidas o con etiquetas contradictorias. Particiones usadas por todos los experimentos: `results/particiones_finales.csv` (train 5.616 · validation 1.878 · test 1.874). Imágenes excluidas y motivo: `results/imagenes_excluidas.csv`. Detalle en `docs/dataset.md`.

## Entorno
- Google Colab con GPU **Tesla T4** (16 GB) · Python 3.13
- Bibliotecas y versiones exactas: `requirements.txt` (PyTorch, torchvision, scikit-learn, pandas, numpy, matplotlib, Pillow, imagehash, tqdm)

## Cómo ejecutar
1. Crear un token de GitHub (classic, permiso `repo`) y guardarlo en Colab como secreto `GH_TOKEN` (ícono de llave → Acceso al notebook).
2. Subir el zip de datos a `Mi unidad/yuca_datos/` en Google Drive.
3. Abrir cada notebook en Colab con GPU T4, poner el nombre y correo propios en la primera celda y ejecutar las celdas en orden.

| Orden | Notebook | Qué hace | Requiere |
|---|---|---|---|
| 1 | `notebooks/00_entorno.ipynb` | Entorno, descarga y verificación de los datos | — |
| 2 | `notebooks/01_eda_duplicados.ipynb` | EDA, duplicados (MD5 y pHash) y particiones finales | 00 |
| 3 | `notebooks/02_E0_baseline.ipynb` | E0 y código común de datos y métricas | 01 |
| 4 | `notebooks/03_E1_feature_extraction.ipynb` | E1 (prueba de 1 época y 10 épocas) | 02 |
| 5 | `notebooks/04_E2_finetuning.ipynb` | E2, partiendo de `checkpoints/E1.pt` | 03 |
| 6 | `notebooks/05_E3_resnet18.ipynb` | E3 y medición de recursos E1 vs E3 | 03 |
| 7 | `notebooks/06_E4_augmentation.ipynb` | E4 y tabla consolidada | 05 |
| 8 | `notebooks/08_analisis_errores.ipynb` | Análisis de errores, Grad-CAM y pruebas de brillo y forma | 04 a 07 |
| — | `notebooks/07_demo_inferencia.ipynb` | Demostración en vivo (inferencia y recálculo de macro-F1) | 03 |

## Reproducibilidad
- Semilla **42** en todos los experimentos (`src/utils.py`). En GPU el determinismo no es total: dos ejecuciones de E1 dieron macro-F1 de 0,6944 y 0,7022; los valores oficiales son los guardados en `results/*/metricas.json`.
- Preprocesamiento: lado corto a 256 + recorte central 224 + normalización de ImageNet (`src/data.py`).
- Pesos entrenados en `checkpoints/` (E1, E2, E3, E4). `results/E1_prueba/` es la prueba de 1 época usada en el control del día 7; no se usa en las comparaciones.

## Estructura
```
scripts/      carga y verificación de datos
src/          código común: datos, modelos, entrenamiento, métricas
notebooks/    un notebook por etapa y experimento
results/      métricas, curvas, matrices, predicciones y análisis de errores
checkpoints/  pesos de los modelos
docs/         dataset.md, protocolo.md, hipotesis.md, bitacora.md, declaración de IA y evidencias/
```

## Documentación
- Protocolo y decisiones: `docs/protocolo.md`
- Hipótesis previas: `docs/hipotesis.md`
- Bitácora cronológica: `docs/bitacora.md`
- Capturas de cada paso: `docs/evidencias/`
- Uso de inteligencia artificial: `docs/declaracion_ia.md`

## Referencia del conjunto de datos
Mwebaze, E., Gebru, T., Frome, A., Nsumba, S., & Tusubira, J. (2019). *iCassava 2019 Fine-Grained Visual Categorization Challenge*. arXiv:1908.02900.
