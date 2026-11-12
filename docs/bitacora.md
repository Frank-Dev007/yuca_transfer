# Bitácora del proyecto

## 2026-11-12 — Frank

### 1. Entorno de ejecución
**Problema:** Elegir framework y dónde ejecutar.
**Alternativas:** PyTorch o Keras; PC local (RTX 3050 4 GB) o Google Colab (T4).
**Decisión:** PyTorch en Google Colab (GPU Tesla T4).
**Prueba:** Clonación del repositorio y nvidia-smi en Colab.
**Resultado:** Entorno funcionando.
**Revisión de la decisión:** (completar: por qué Colab y no el PC)

### 2. Acceso a GitHub desde Colab
**Problema:** Clonar y subir cambios a un repositorio privado desde Colab.
**Alternativas:** Repositorio público o token personal guardado como secreto de Colab.
**Decisión:** Token fine-grained (solo yuca_transfer, Contents: read/write) guardado como secreto GH_TOKEN.
**Prueba:** Clonación y push desde Colab.
**Resultado:** (completar)
**Revisión de la decisión:** (completar)

### 3. Obtención de los datos
**Problema:** Cargar el dataset en Colab.
**Alternativas:** Descarga automática por script o carga manual desde Google Drive.
**Decisión:** Carga manual del zip a Drive y copia a la sesión con scripts/preparar_datos.py.
**Prueba:** scripts/verificar_datos.py.
**Resultado:** (completar con lo que mostró la Celda 7)
**Revisión de la decisión:** (completar: motivo de preferir la carga manual)

### Consultas a IA
- Consulta: qué framework y entorno usar; cómo configurar Colab y GitHub; cómo cargar y verificar los datos.
- Respuesta recibida (resumen): sugerencia de PyTorch y Colab; código de configuración y verificación de datos.
- Decisión propia: (completar)
