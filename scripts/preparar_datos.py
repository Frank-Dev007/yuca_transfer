import shutil, zipfile
from pathlib import Path

ZIP_DRIVE = Path("/content/drive/MyDrive/yuca_datos/cassavaleafdata.zip")
DATA_DIR = Path("data")
ZIP_LOCAL = DATA_DIR / ZIP_DRIVE.name

DATA_DIR.mkdir(exist_ok=True)
if not ZIP_LOCAL.exists():
    print("Copiando desde Drive...")
    shutil.copy(ZIP_DRIVE, ZIP_LOCAL)
print("Tamaño del zip (MB):", round(ZIP_LOCAL.stat().st_size / 1e6, 1))

with zipfile.ZipFile(ZIP_LOCAL) as z:
    z.extractall(DATA_DIR)
print("Contenido de data/:", [p.name for p in DATA_DIR.iterdir()])
