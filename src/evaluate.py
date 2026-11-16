import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from src.data import CLASES

def calcular_metricas(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=range(5), target_names=CLASES,
                                output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(5), zero_division=0)),
        "por_clase": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "soporte": int(rep[c]["support"])} for c in CLASES},
    }

def imprimir_metricas(m, nombre):
    print(f"{nombre}: accuracy = {m['accuracy']:.4f} | macro-F1 = {m['macro_f1']:.4f}")
    for c, v in m["por_clase"].items():
        print(f"   {c:8s} recall = {v['recall']:.3f}   F1 = {v['f1']:.3f}   (n = {v['soporte']})")

def guardar_matriz_confusion(y_true, y_pred, ruta, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.show()
    return cm

def guardar_json(datos, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
