import json
import numpy as np
import pandas as pd
from pathlib import Path

dossier = Path("data/processed")

# Charger les données
X = np.load(dossier / "X.npy")
meta = pd.read_csv(dossier / "meta.csv")

# Définir l'ordre EXACT des classes (10 classes)
classes = [
    "Miarahaba", "Eny", "Tsia", "Miezaka", "Maheno",
    "Marenina", "Mora", "Tanora", "Solosaina", "Maty"
]

# Recréer les labels en fonction du nom du signe dans meta
mapping = {signe: i for i, signe in enumerate(classes)}
y = np.array([mapping[signe] for signe in meta["signe"]])

# Sauvegarder
np.save(dossier / "y.npy", y)
meta["label"] = y
meta.to_csv(dossier / "meta.csv", index=False)

with open(dossier / "classes.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print(f"Labels corrigés pour {len(classes)} classes.")
print(f"y shape : {y.shape}, valeurs uniques : {np.unique(y)}")