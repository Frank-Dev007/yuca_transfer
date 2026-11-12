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
- **Resolución:** (se medirá en el EDA)
- **Procedencia:** fotografías de campo; país y condiciones por verificar en el artículo de referencia.
- **Criterios de exclusión:** (definir tras el EDA)
- **Duplicados encontrados:** (se completará en el EDA)

## Referencia
Mwebaze, E., Gebru, T., Frome, A., Nsumba, S., & Tusubira, J. (2019).
iCassava 2019 Fine-Grained Visual Categorization Challenge. arXiv:1908.02900.
