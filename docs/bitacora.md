# Bitácora del proyecto

## 2026-11-12 — Frank

### 1. Entorno de ejecución
**Problema:** Elegir framework y dónde ejecutar.
**Alternativas:** PyTorch o Keras; PC local (RTX 3050 4 GB) o Google Colab (T4).
**Decisión:** PyTorch en Google Colab (GPU Tesla T4).
**Prueba:** Clonación del repositorio y nvidia-smi en Colab.
**Resultado:** Entorno funcionando.
**Revisión de la decisión:** Elegí Colab porque en mi PC tengo Python 3.14 y no quería cambiar de versión, y mi GPU solo tiene 4 GB de memoria. En Colab ya viene todo instalado y los tres trabajamos en el mismo entorno. El riesgo es que la sesión se corte o que no haya GPU disponible el día del control, y hay que tenerlo en cuenta.

### 2. Acceso a GitHub desde Colab
**Problema:** Clonar y subir cambios a un repositorio privado desde Colab.
**Alternativas:** Repositorio público o token personal guardado como secreto de Colab.
**Decisión:** Token fine-grained (solo yuca_transfer, Contents: read/write) guardado como secreto GH_TOKEN.
**Prueba:** Clonación y push desde Colab.
**Resultado:** La clonación y el push funcionan con el token guardado como secreto. Los primeros commits salieron con el usuario Franco-DLC porque el correo configurado en git no era el de mi cuenta; lo corregí y ahora salen a nombre de Frank-Dev007
**Revisión de la decisión:** Se mantiene, pero cambié la forma de manejar el token. En el primer intento lo pegué directamente en el código y quedó visible, así que lo eliminé y creé uno nuevo, guardándolo solo en los secretos de Colab. También aprendí que el correo de git debe ser el mismo de la cuenta de GitHub.

### 3. Obtención de los datos
**Problema:** Cargar el dataset en Colab.
**Alternativas:** Descarga automática por script o carga manual desde Google Drive.
**Decisión:** Carga manual del zip a Drive y copia a la sesión con scripts/preparar_datos.py.
**Prueba:** scripts/verificar_datos.py.
**Resultado:** Las tres particiones coinciden con las cifras oficiales: train 5656, validation 1889 y test 1885, con 5 clases cada una. Las clases están desbalanceadas: cmd tiene el 47 % de las imágenes y healthy solo el 5,6 %, con las mismas proporciones en las tres particiones.
**Revisión de la decisión:** Preferí descargar el zip yo mismo y subirlo a Drive para evitar complicaciones después: el archivo ya queda guardado y solo se copia a Colab en cada sesión. Como el profesor no tiene acceso a mi Drive, dejé el enlace oficial de descarga en dataset.md.

## 2026-11-13 — Frank

### 4. Control de duplicados y exclusión de imágenes
**Problema:** Verificar si había imágenes repetidas entre train, validation y test (fuga de datos), como pide el taller.
**Alternativas:** Para detectarlas: MD5 (copias exactas) y pHash con distintos umbrales (casi duplicados). Para tratarlas: quitar las copias de train, quitar las de test o no tocar nada.
**Decisión:** Usamos MD5 y pHash con umbral 0. Decidimos en equipo quitar de train las copias de imágenes que también están en test o validation, y excluir de todas las particiones los grupos de duplicados con etiquetas distintas.
**Prueba:** Notebook 01_eda_duplicados.ipynb (Celdas 6 a 12).
**Resultado:** 26 grupos de copias exactas (68 imágenes), 17 cruzan particiones y 18 tienen etiquetas distintas. pHash encontró 4 casi duplicados más. Se excluyeron 62 imágenes y las particiones finales quedaron en train 5616, validation 1878 y test 1874. Comprobación final: 0 copias entre particiones.
**Revisión de la decisión:** Se mantiene. Quitamos las copias de train porque el test es como el examen y debe quedar fijo para todos los experimentos. Las imágenes con etiquetas contradictorias las sacamos también de test porque no hay forma de saber cuál etiqueta es la correcta, y evaluar con una respuesta dudosa no sería justo. Lo que perdemos es que el test queda con 11 imágenes menos que el oficial, y lo vamos a mencionar como limitación.


### Consultas a IA
- Consulta: qué framework y entorno usar; cómo configurar Colab y GitHub; cómo cargar y verificar los datos.
- Respuesta recibida (resumen): sugerencia de PyTorch y Colab; código de configuración y verificación de datos.
- Decisión propia: (completar)
