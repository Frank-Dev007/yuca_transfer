import torch.nn as nn
from torchvision import models

def crear_modelo(nombre, n_clases=5):
    """Backbone preentrenado en ImageNet, totalmente congelado, con una cabeza lineal nueva entrenable."""
    if nombre == "mobilenet_v2":
        modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in modelo.parameters():
            p.requires_grad = False
        modelo.classifier = nn.Linear(modelo.last_channel, n_clases)   # 1280 -> 5
    elif nombre == "resnet18":
        modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in modelo.parameters():
            p.requires_grad = False
        modelo.fc = nn.Linear(modelo.fc.in_features, n_clases)        # 512 -> 5
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")
    return modelo

def descongelar_ultimo_bloque(modelo, nombre):
    """E2: vuelve entrenable el bloque final del backbone."""
    if nombre == "mobilenet_v2":
        bloque = modelo.features[-2:]   # último bloque residual invertido + convolución final
    elif nombre == "resnet18":
        bloque = modelo.layer4
    for p in bloque.parameters():
        p.requires_grad = True
    return modelo
