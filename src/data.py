import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]
CLASE_A_IDX = {c: i for i, c in enumerate(CLASES)}
MEDIA = [0.485, 0.456, 0.406]   # normalización de ImageNet (protocolo)
DESV = [0.229, 0.224, 0.225]

# Decisión 3 del protocolo: lado corto a 256 y recorte central de 224
transform_base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
transform_modelo = transforms.Compose([transform_base, transforms.ToTensor(),
                                       transforms.Normalize(MEDIA, DESV)])

# E4 (experimento adicional): augmentation moderado SOLO para train.
# Sin cambios de color, para no alterar el aspecto de los síntomas.
transform_aumentado = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESV),
])

class YucaDataset(Dataset):
    def __init__(self, tabla, transform):
        self.rutas = tabla["ruta"].tolist()
        self.etiquetas = [CLASE_A_IDX[c] for c in tabla["clase"]]
        self.transform = transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        img = Image.open(self.rutas[i]).convert("RGB")
        return self.transform(img), self.etiquetas[i]

def cargar_particiones(csv="results/particiones_finales.csv"):
    df = pd.read_csv(csv)
    return {s: df[df.split == s].reset_index(drop=True) for s in ["train", "validation", "test"]}

def crear_dataloaders(batch_size=32, seed=42, transform=transform_modelo,
                      transform_train=None, num_workers=2):
    particiones = cargar_particiones()
    generador = torch.Generator().manual_seed(seed)
    loaders = {}
    for split, tabla in particiones.items():
        t = transform_train if (split == "train" and transform_train is not None) else transform
        loaders[split] = DataLoader(
            YucaDataset(tabla, t), batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generador if split == "train" else None,
            num_workers=num_workers, pin_memory=True)
    return loaders
