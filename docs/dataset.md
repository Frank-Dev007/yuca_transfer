# Ficha del conjunto de datos

- **Nombre:** cassava (iCassava 2019), TensorFlow Datasets
- **Ficha oficial:** https://www.tensorflow.org/datasets/catalog/cassava (consultada el 2026-11-12)
- **Archivo:** https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip (descargado el 2026-11-12)
- **Carga en Colab:** zip subido manualmente a Google Drive (MyDrive/yuca_datos) y copiado a la sesión con scripts/preparar_datos.py
- **Origen (homepage):** https://www.kaggle.com/c/cassava-disease/overview — competencia iCassava 2019
- **Versión:** 0.1.0
- **Licencia:** no especificada en la ficha de TFDS (la licencia CC BY 4.0 del pie de página corresponde al contenido del sitio web, no a las imágenes). Pendiente: verificar en la página de la competencia.
- **Clases (5):** cbb (plaga bacteriana), cbsd (raya marrón), cgm (ácaro verde), cmd (mosaico), healthy (sana)
- **Particiones oficiales:** train 5.656 · validation 1.889 · test 1.885 · total 9.430

## Distribución verificada (scripts/verificar_datos.py)

| Clase   | Train        | Validation  | Test        |
|---------|--------------|-------------|-------------|
| cbb     | 466 (8,2 %)  | 156 (8,3 %) | 155 (8,2 %) |
| cbsd    | 1443 (25,5 %)| 482 (25,5 %)| 481 (25,5 %)|
| cgm     | 773 (13,7 %) | 258 (13,7 %)| 258 (13,7 %)|
| cmd     | 2658 (47,0 %)| 887 (47,0 %)| 886 (47,0 %)|
| healthy | 316 (5,6 %)  | 106 (5,6 %) | 105 (5,6 %) |
| **Total** | **5656**   | **1889**    | **1885**    |

- **Formato:** imágenes RGB de tamaño variable.
- **Resolución:** tamaño variable, entre 420×347 y 1622×1240 píxeles (mediana 543×500). Tamaños más frecuentes: 500×666 (2369), 888×500 (1328), 625×500 (1073), 666×500 (1070), 500×500 (719), 500×888 (512). Mediana por clase: cbb 500×500, cbsd 625×500, cgm 500×584, cmd 507×500, healthy 625×500.

- **Procedencia:** fotografías de campo; país y condiciones por verificar en el artículo de referencia.
- **Criterios de exclusión:** ver docs/protocolo.md, decisiones 1 y 2. Se excluyeron 62 imágenes: 56 por pertenecer a grupos de duplicados con etiquetas contradictorias (35 de train, 10 de validation, 11 de test) y 6 por ser copias de imágenes de otra partición (5 de train, 1 de validation). Particiones finales: train 5616, validation 1878, test 1874. Lista completa en results/imagenes_excluidas.csv; particiones finales en results/particiones_finales.csv.

- **Duplicados encontrados:**
    - Copias exactas (MD5): 26 grupos con 68 imágenes. 17 grupos cruzan particiones y 18 tienen etiquetas distintas (13 de ellos entre particiones). En los 20 grupos más grandes, 15 tienen una copia como cbb, otra como cbsd y otra como cgm.
    - Test con copia exacta en train: 11; en validation: 4. Validation con copia exacta en train: 12.
    - pHash (umbral 0): 36 pares entre particiones; 32 coinciden con MD5 y 4 son casi idénticos. 30 de los 36 pares tienen etiquetas distintas.
    - Detalle en results/grupos_duplicados_exactos.csv, results/duplicados_exactos.csv y results/duplicados_phash_umbral0.csv; figuras en docs/evidencias.

## Referencia
Mwebaze, E., Gebru, T., Frome, A., Nsumba, S., & Tusubira, J. (2019).
iCassava 2019 Fine-Grained Visual Categorization Challenge. arXiv:1908.02900.
