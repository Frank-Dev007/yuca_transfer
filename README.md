# Clasificación de enfermedades en hojas de yuca

Repositorio del trabajo de **Aprendizaje Computacional**. Está en construcción y se irá actualizando a medida que avance el proyecto. Por ahora reúne el código, los notebooks y las evidencias de lo que llevamos, pero todavía faltan experimentos, ajustes y documentación.

La fuente principal de avance es [`bitacora_proyecto.md`](bitacora_proyecto.md). Allí se registra, por fecha e integrante, qué problema se abordó, qué alternativas había, qué se decidió, cómo se probó, qué resultado se obtuvo y si la decisión se mantiene o se cambia. Ese es el documento que mejor refleja el estado real del trabajo.

## Entorno

- Framework: **PyTorch**.
- Entorno de ejecución: **Google Colab con GPU Tesla T4**.
- Repositorio privado: `yuca_transfer`.
- Acceso desde Colab con un token fine-grained guardado como secreto `GH_TOKEN` (permiso `Contents: read/write` sobre `yuca_transfer`).

## Configuración rápida en Colab

1. Guardar el token de GitHub como secreto `GH_TOKEN` en Colab.
2. Clonar el repositorio privado.
3. Copiar el dataset desde Google Drive a la sesión con `scripts/preparar_datos.py`.
4. Verificar las particiones y las clases con `scripts/verificar_datos.py`.

## Qué hay hasta ahora

El proyecto busca clasificar imágenes de hojas de yuca en 5 clases (`cbb`, `cbsd`, `cgm`, `cmd` y `healthy`) usando PyTorch. Trabajamos en Google Colab con GPU Tesla T4 y el repositorio privado `yuca_transfer`, al que se accede desde Colab con un token guardado como secreto.

El dataset tiene las particiones oficiales train 5656, validation 1889 y test 1885, con las cinco clases y un desbalance claro: `cmd` concentra el 47 % de las imágenes y `healthy` solo el 5,6 %. Después de revisar duplicados y copias entre particiones, las particiones de trabajo quedaron en train 5616, validation 1878 y test 1874. El enlace oficial de descarga está en [`dataset.md`](dataset.md); las imágenes no se suben al repositorio.

Hasta el momento se han hecho dos experimentos. E0 es un baseline con regresión logística sobre histogramas de color, que sirvió para ver qué tan lejos se llega sin una arquitectura sofisticada. E1 usa MobileNetV2 preentrenada en ImageNet y congelada, con una cabeza lineal, y mejoró bastante frente a E0: el macro-F1 en test pasó de 0,2997 a 0,7022 en 10 épocas. También se hizo una prueba parcial de E1 con 1 época para verificar que el pipeline funcionaba antes de lanzar el entrenamiento completo.

Los resultados, curvas, matrices de confusión y demás evidencias se van guardando en el repositorio y se referencian en la bitácora. No se presenta aquí una estructura cerrada de carpetas porque todavía se están agregando notebooks, scripts y resultados, y preferimos que la bitácora y el propio repositorio muestren el avance real.

## Lo que falta

Todavía queda bastante por hacer. Entre otras cosas, falta seguir probando modelos y configuraciones, revisar con más detalle las clases que siguen difíciles (sobre todo `cbb`), decidir si conviene descongelar parte de la red o afinar el backbone, y documentar cada decisión en la bitácora. También faltan ajustes de organización del repositorio, revisiones de resultados y las demás actividades del curso. Este README se irá completando cuando esas partes estén más claras.

## Equipo

- Frank
- Juan
- Tomas
