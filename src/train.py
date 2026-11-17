import os, time
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.data import CLASES
from src.evaluate import calcular_metricas, imprimir_metricas, guardar_matriz_confusion, guardar_json

def poner_modo_entrenamiento(modelo):
    """Modo entrenamiento, pero las BatchNorm congeladas se quedan en modo evaluación
    para que el backbone congelado no cambie sus estadísticas internas."""
    modelo.train()
    for capa in modelo.modules():
        if isinstance(capa, nn.BatchNorm2d) and not any(p.requires_grad for p in capa.parameters()):
            capa.eval()

def predecir(modelo, loader, dispositivo):
    modelo.eval()
    reales, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            salida = modelo(x.to(dispositivo, non_blocking=True))
            preds.append(salida.argmax(1).cpu())
            reales.append(y)
    return torch.cat(reales).numpy(), torch.cat(preds).numpy()

def entrenar(modelo, loaders, dispositivo, epocas, lr, ruta_checkpoint):
    modelo.to(dispositivo)
    optimizador = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad], lr=lr)
    criterio = nn.CrossEntropyLoss()
    historial, mejor_f1, mejor_epoca = [], -1.0, 0
    t0 = time.time()
    for epoca in range(1, epocas + 1):
        # Entrenamiento
        poner_modo_entrenamiento(modelo)
        suma_perdida, aciertos, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
            optimizador.zero_grad()
            salida = modelo(x)
            perdida = criterio(salida, y)
            perdida.backward()
            optimizador.step()
            suma_perdida += perdida.item() * len(y)
            aciertos += (salida.argmax(1) == y).sum().item()
            n += len(y)
        # Validación
        modelo.eval()
        suma_val, n_val, reales, preds = 0.0, 0, [], []
        with torch.no_grad():
            for x, y in loaders["validation"]:
                x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
                salida = modelo(x)
                suma_val += criterio(salida, y).item() * len(y)
                n_val += len(y)
                preds.append(salida.argmax(1).cpu())
                reales.append(y.cpu())
        m = calcular_metricas(torch.cat(reales).numpy(), torch.cat(preds).numpy())
        fila = {"epoca": epoca,
                "perdida_train": suma_perdida / n, "accuracy_train": aciertos / n,
                "perdida_val": suma_val / n_val, "accuracy_val": m["accuracy"],
                "macro_f1_val": m["macro_f1"], "segundos_acumulados": time.time() - t0}
        historial.append(fila)
        print(f"Época {epoca:2d} | pérdida train {fila['perdida_train']:.4f} | "
              f"pérdida val {fila['perdida_val']:.4f} | macro-F1 val {fila['macro_f1_val']:.4f} | "
              f"{fila['segundos_acumulados']:.0f} s")
        if m["macro_f1"] > mejor_f1:              # decisión 11: mejor macro-F1 en validación
            mejor_f1, mejor_epoca = m["macro_f1"], epoca
            torch.save(modelo.state_dict(), ruta_checkpoint)
    tiempo = time.time() - t0
    modelo.load_state_dict(torch.load(ruta_checkpoint, map_location=dispositivo))
    return pd.DataFrame(historial), {"mejor_epoca": mejor_epoca, "mejor_macro_f1_val": mejor_f1,
                                     "tiempo_entrenamiento_s": tiempo}

def medir_inferencia_ms(modelo, loader, dispositivo, n_lotes=20):
    """Tiempo del modelo por imagen (lotes de 32, sin contar la carga de imágenes)."""
    modelo.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(dispositivo)
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            modelo(x)
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
            if i > 0:                          # el primer lote se descarta (calentamiento)
                total += time.time() - t
                n += len(x)
            if i >= n_lotes:
                break
    return total / n * 1000

def graficar_curvas(historial, ruta, titulo):
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(historial.epoca, historial.perdida_train, "o-", label="train")
    ejes[0].plot(historial.epoca, historial.perdida_val, "o-", label="validación")
    ejes[0].set_title("Pérdida")
    ejes[0].set_xlabel("Época")
    ejes[0].legend()
    ejes[1].plot(historial.epoca, historial.accuracy_train, "o-", label="accuracy train")
    ejes[1].plot(historial.epoca, historial.accuracy_val, "o-", label="accuracy validación")
    ejes[1].plot(historial.epoca, historial.macro_f1_val, "o-", label="macro-F1 validación")
    ejes[1].set_title("Métricas")
    ejes[1].set_xlabel("Época")
    ejes[1].legend()
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.show()

def ejecutar_experimento(nombre, modelo, loaders, dispositivo, epocas, lr, semilla, descripcion, extra=None):
    carpeta = f"results/{nombre}"
    os.makedirs(carpeta, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    ruta_ckpt = f"checkpoints/{nombre}.pt"

    historial, info = entrenar(modelo, loaders, dispositivo, epocas, lr, ruta_ckpt)
    historial.to_csv(f"{carpeta}/historial.csv", index=False)
    graficar_curvas(historial, f"{carpeta}/curvas.png", f"{nombre} · {descripcion}")

    y_val, p_val = predecir(modelo, loaders["validation"], dispositivo)
    y_test, p_test = predecir(modelo, loaders["test"], dispositivo)
    m_val, m_test = calcular_metricas(y_val, p_val), calcular_metricas(y_test, p_test)
    imprimir_metricas(m_val, "Validación (mejor época)")
    print()
    imprimir_metricas(m_test, "Test")
    guardar_matriz_confusion(y_test, p_test, f"{carpeta}/matriz_confusion_test.png",
                             f"{nombre} · {descripcion} · Test")

    # Predicciones de test, para el análisis de errores
    pd.DataFrame({"ruta": loaders["test"].dataset.rutas,
                  "real": [CLASES[i] for i in y_test],
                  "predicha": [CLASES[i] for i in p_test]}).to_csv(f"{carpeta}/predicciones_test.csv", index=False)

    resumen = {"experimento": nombre, "descripcion": descripcion, "semilla": semilla,
               "epocas": epocas, "lr": lr, "batch_size": loaders["train"].batch_size, **info,
               "validacion": m_val, "test": m_test,
               "tiempo_inferencia_ms_por_imagen": medir_inferencia_ms(modelo, loaders["test"], dispositivo),
               "parametros_totales": sum(p.numel() for p in modelo.parameters()),
               "parametros_entrenables": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
               "tamano_checkpoint_mb": os.path.getsize(ruta_ckpt) / 1e6,
               "gpu": torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "cpu",
               **(extra or {})}
    guardar_json(resumen, f"{carpeta}/metricas.json")
    print(f"\nMejor época: {info['mejor_epoca']} | entrenamiento: {info['tiempo_entrenamiento_s']:.0f} s | "
          f"inferencia: {resumen['tiempo_inferencia_ms_por_imagen']:.2f} ms/imagen | "
          f"parámetros entrenables: {resumen['parametros_entrenables']:,}")
    return resumen
