# Bitácora del proyecto

## 2026-11-12 — Frank

### 1. Entorno de ejecución

**Problema:** Elegir framework y dónde ejecutar.  
**Alternativas:** PyTorch o Keras; PC local (RTX 3050 4 GB) o Google Colab (T4).  
**Decisión:** PyTorch en Google Colab (GPU Tesla T4).  
**Prueba:** Clonación del repositorio y `nvidia-smi` en Colab.  
**Resultado:** Entorno funcionando.  
**Revisión de la decisión:** Elegí Colab porque en mi PC tengo Python 3.14 y no quería cambiar de versión, y mi GPU solo tiene 4 GB de memoria. En Colab ya viene todo instalado y los tres trabajamos en el mismo entorno. El riesgo es que la sesión se corte o que no haya GPU disponible el día del control, y hay que tenerlo en cuenta.

### 2. Acceso a GitHub desde Colab

**Problema:** Clonar y subir cambios a un repositorio privado desde Colab.  
**Alternativas:** Repositorio público o token personal guardado como secreto de Colab.  
**Decisión:** Token fine-grained (solo `yuca_transfer`, Contents: read/write) guardado como secreto `GH_TOKEN`.  
**Prueba:** Clonación y push desde Colab.  
**Resultado:** La clonación y el push funcionan con el token guardado como secreto. Los primeros commits salieron con el usuario Franco-DLC porque el correo configurado en git no era el de mi cuenta; lo corregí y ahora salen a nombre de Frank-Dev007.  
**Revisión de la decisión:** Se mantiene, pero cambié la forma de manejar el token. En el primer intento lo pegué directamente en el código y quedó visible, así que lo eliminé y creé uno nuevo, guardándolo solo en los secretos de Colab. También aprendí que el correo de git debe ser el mismo de la cuenta de GitHub.

### 3. Obtención de los datos

**Problema:** Cargar el dataset en Colab.  
**Alternativas:** Descarga automática por script o carga manual desde Google Drive.  
**Decisión:** Carga manual del zip a Drive y copia a la sesión con `scripts/preparar_datos.py`.  
**Prueba:** `scripts/verificar_datos.py`.  
**Resultado:** Las tres particiones coinciden con las cifras oficiales: train 5656, validation 1889 y test 1885, con 5 clases cada una. Las clases están desbalanceadas: cmd tiene el 47 % de las imágenes y healthy solo el 5,6 %, con las mismas proporciones en las tres particiones.  
**Revisión de la decisión:** Preferí descargar el zip yo mismo y subirlo a Drive para evitar complicaciones después: el archivo ya queda guardado y solo se copia a Colab en cada sesión. Como el profesor no tiene acceso a mi Drive, dejé el enlace oficial de descarga en `dataset.md`.

---

## 2026-11-13 — Frank

### 4. Control de duplicados y exclusión de imágenes

**Problema:** Verificar si había imágenes repetidas entre train, validation y test (fuga de datos), como pide el taller.  
**Alternativas:** Para detectarlas: MD5 (copias exactas) y pHash con distintos umbrales (casi duplicados). Para tratarlas: quitar las copias de train, quitar las de test o no tocar nada.  
**Decisión:** Usamos MD5 y pHash con umbral 0. Decidimos en equipo quitar de train las copias de imágenes que también están en test o validation, y excluir de todas las particiones los grupos de duplicados con etiquetas distintas.  
**Prueba:** Notebook `01_eda_duplicados.ipynb` (Celdas 6 a 12).  
**Resultado:** 26 grupos de copias exactas (68 imágenes), 17 cruzan particiones y 18 tienen etiquetas distintas. pHash encontró 4 casi duplicados más. Se excluyeron 62 imágenes y las particiones finales quedaron en train 5616, validation 1878 y test 1874. Comprobación final: 0 copias entre particiones.  
**Revisión de la decisión:** Se mantiene. Quitamos las copias de train porque el test es como el examen y debe quedar fijo para todos los experimentos. Las imágenes con etiquetas contradictorias las sacamos también de test porque no hay forma de saber cuál etiqueta es la correcta, y evaluar con una respuesta dudosa no sería justo. Lo que perdemos es que el test queda con 11 imágenes menos que el oficial, y lo vamos a mencionar como limitación.

### Consultas a IA

- **Consulta:** qué framework y entorno usar; cómo configurar Colab y GitHub; cómo cargar y verificar los datos.
- **Respuesta recibida (resumen):** sugerencia de PyTorch y Colab; código de configuración y verificación de datos.
- **Decisión propia:** (completar)

---

## 2026-11-16 — Juan

Configuración de mi entorno en Colab (T4) y acceso al repositorio y a los datos.

### 5. E0 baseline

**Problema:** Tener un primer modelo que nos permita saber qué tan bien se pueden clasificar las imágenes de las hojas de yuca y utilizar sus resultados como referencia para comparar con los siguientes modelos.  
**Decisión:** Utilizar regresión logística trabajando con histogramas de color (32 intervalos por cada uno de los 3 canales). También se aplicó StandardScaler para normalizar los datos y ayudar al modelo durante el entrenamiento.  
**Prueba:** Notebook `02_E0_baseline.ipynb`.  
**Resultado:** En las pruebas se obtuvo un accuracy de 0,5491 y un macro-F1 de 0,2997. El recall por clase fue: cbb 0,110; cbsd 0,419; cgm 0,032; cmd 0,901 y healthy 0,067. El tiempo de entrenamiento fue de 2,45 segundos y la extracción de características tardó 9,5 ms por imagen.  
**Revisión de la decisión:** La hipótesis se cumplió parcialmente, porque el modelo sí logró reconocer algunas de las clases, pero tuvo bastantes dificultades con otras. Por ejemplo, reconoció mejor la clase cmd, mientras que para cgm y healthy los resultados fueron muy bajos. Esto puede deberse a que el modelo solamente tiene en cuenta los colores y no otros detalles de las hojas que ayudan a diferenciar una enfermedad de otra. También pudo influir que algunas clases tengan más imágenes que otras, haciendo que el modelo se inclinara más por predecir cmd.

**¿Qué se consigue sin una arquitectura sofisticada?**  
Con este primer modelo conseguimos tener una idea de cómo se comportan los datos y qué tan difícil es clasificar las hojas de yuca. Aunque no se obtuvieron resultados muy buenos en todas las clases, nos sirvió para comprobar que se pueden reconocer algunas diferencias usando solamente los colores. También nos permitió darnos cuenta de que necesitamos un modelo que pueda analizar más detalles de las imágenes y no quedarse únicamente con sus colores.

### 6. E1 · Extracción de características (MobileNetV2 congelada + cabeza lineal)

#### 6.1 Prueba de 1 época (E1 parcial)

**Problema:** Verificar que el pipeline de entrenamiento (`models.py`, `train.py`) funciona antes de lanzar las 10 épocas de E1.  
**Decisión:** Prueba de 1 época con MobileNetV2 congelada + cabeza lineal, con la misma configuración del protocolo (semilla 42, batch 32, Adam, LR 0,001).  
**Prueba:** Ejecución en Colab con `src/models.py` y `src/train.py`. Resultados guardados en `results/E1_prueba/` y `checkpoints/E1_prueba.pt`. El notebook de E1 se subirá al repositorio junto con E1 completo.  
**Resultado:** La época tardó 60 s. Pérdida train 0,8776 y validación 0,7157. Validación: accuracy 0,7444, macro-F1 0,6297. Test: accuracy 0,7481, macro-F1 0,6125. Recall en test: cbb 0,279 · cbsd 0,807 · cgm 0,470 · cmd 0,921 · healthy 0,381. Evidencias: `docs/evidencias/12_E1_prueba_curvas.png` y `13_E1_prueba_metricas_matriz.png`.  
**Revisión de la decisión:** Se mantiene. La prueba corta sirvió para comprobar que el entrenamiento, la evaluación y el guardado de resultados funcionan sin errores antes de gastar tiempo de GPU en las 10 épocas. Como cada época tardó unos 60 s, calculamos que E1 completo tomaría alrededor de 10 minutos, lo que cabía en la sesión de Colab, así que seguimos con la ejecución completa. Esta prueba también cumple el requisito de una ejecución parcial de E1 para el control del día 7. Los resultados de 1 época no se usarán para comparar con los demás experimentos; para eso se usará E1 completo.

## 2026-11-17 — Juan

#### 6.2 E1 completo (10 épocas) 

**Problema:** Ejecutar E1 con el presupuesto completo del protocolo (10 épocas) y elegir el modelo que se evalúa en test.  
**Alternativas:** Evaluar la última época o la época con mejor macro-F1 en validación (decisión 11 del protocolo).  
**Decisión:** Misma configuración de la prueba (semilla 42, batch 32, Adam, LR 0,001), 10 épocas, guardando la época con mejor macro-F1 en validación.  
**Prueba:** `notebooks/03_E1_feature_extraction.ipynb`. El entrenamiento se ejecutó el 2026-11-16 y sus resultados se subieron ese mismo día (`results/E1/` y `checkpoints/E1.pt`); el notebook se subió el 2026-11-17.  
**Resultado:** Entrenamiento de 543 s en total. Mejor época: 9 (macro-F1 en validación 0,7094). En la época 10 la pérdida de validación subió a 0,6578 y el macro-F1 bajó a 0,6936. Validación (mejor época): accuracy 0,7865, macro-F1 0,7094. Test: accuracy 0,7924, macro-F1 0,7022. Recall en test: cbb 0,377 · cbsd 0,774 · cgm 0,763 · cmd 0,897 · healthy 0,676. Frente a E0 (macro-F1 en test 0,2997), el macro-F1 de E1 es 0,4025 mayor. Evidencia: `docs/evidencias/14_E1_completo_10_epocas.png`.  
**Revisión de la decisión:** Se mantiene. El criterio de guardar la mejor época en validación evitó usar la época 10, en la que el modelo empeoró en validación. La predicción medible de la hipótesis de E1 (macro-F1 en test mayor que E0) se cumplió. Creemos que la mejora frente a E0 se debe a que MobileNetV2, al estar preentrenada en ImageNet, ya trae filtros que describen texturas, bordes y formas de las lesiones, y no solo el color; por eso las clases que E0 casi no reconocía subieron mucho (cgm de 0,032 a 0,763, healthy de 0,067 a 0,676 y cbb de 0,110 a 0,377), mientras que cmd se mantuvo alto (0,901 → 0,897). Las curvas muestran que la pérdida de train bajó de forma sostenida desde 0,8776, pero la validación dejó de mejorar en la época 9 y empeoró en la 10 (pérdida de validación 0,6578 y macro-F1 0,6936), lo que indica que con el backbone congelado el modelo empieza a sobreajustar en ese punto y confirma que el criterio de mejor época en validación era el adecuado. cbb sigue siendo la clase más difícil, probablemente porque tiene menos imágenes y lesiones más variadas, y queda como el punto a atacar en los siguientes experimentos.

**¿Qué aporta el aprendizaje por transferencia?** 

El uso de MobileNetV2 preentrenada nos permitió obtener mejores resultados sin tener que entrenar una red desde cero. Mientras E0 alcanzó un macro-F1 de 0,2997, E1 llegó a 0,7022 en solo 10 épocas. Además, mejoró notablemente el reconocimiento de clases como cgm, healthy y cbb, y mantuvo un buen resultado en cmd. También redujo el tiempo de entrenamiento, con un total de 543 segundos. Sin embargo, cbb sigue siendo la clase más difícil y el modelo empezó a empeorar en validación, por lo que sería conveniente ajustar o descongelar algunas capas para mejorar su rendimiento.




##### 6.3 Corrección de nombres de E1 y E1_prueba 

**Problema:** Al revisar `results/*/metricas.json`, encontramos que las carpetas estaban al revés: `results/E1_prueba` tenía la ejecución de 10 épocas y `results/E1` la de 1 época (lo mismo con los checkpoints).  
**Alternativas:** Volver a entrenar con los nombres correctos o renombrar los archivos existentes.  
**Decisión:** Renombrar con `git mv` (sin volver a entrenar) y corregir el campo `"experimento"` dentro de cada `metricas.json`.  
**Prueba:** Verificación del campo `"epocas"` en cada `metricas.json` después del cambio.  
**Resultado:** `results/E1` y `checkpoints/E1.pt` corresponden a las 10 épocas, y `results/E1_prueba` y `checkpoints/E1_prueba.pt` a la prueba de 1 época. Los títulos de las gráficas `curvas.png` y de la matriz de confusión conservan el nombre original con el que se generaron.  
**Revisión de la decisión:** Se mantiene. Renombrar evitó gastar otra sesión de GPU, y el historial de git deja registrado el cambio.
