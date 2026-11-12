from pathlib import Path
from collections import Counter

ROOT = Path("data/cassavaleafdata")
ESPERADO = {"train": 5656, "validation": 1889, "test": 1885}   # cifras oficiales
EXT = {".jpg", ".jpeg", ".png"}

for split, esperado in ESPERADO.items():
    conteo = Counter()
    for clase in sorted(p for p in (ROOT / split).iterdir() if p.is_dir()):
        conteo[clase.name] = sum(1 for f in clase.iterdir() if f.suffix.lower() in EXT)
    total = sum(conteo.values())
    estado = "OK" if total == esperado else "REVISAR"
    print(f"\n{split}: {total} imágenes (esperado {esperado}) -> {estado}")
    for c, n in conteo.items():
        print(f"   {c:10s} {n:5d}  ({n / total:.1%})")
