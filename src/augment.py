"""
Augmentation de donnees sur les squelettes.
Genere des copies avec bruit gaussien et jitter temporel.
Entree  : X_train.npy, y_train.npy
Sortie  : X_train_aug.npy, y_train_aug.npy
"""

import numpy as np
from pathlib import Path

# Charger les donnees d'entrainement
dossier = Path("data/processed")
X_train = np.load(dossier / "X_train.npy")
y_train = np.load(dossier / "y_train.npy")

N, T, D = X_train.shape
print(f"X_train original : {X_train.shape}")

# Parametres d'augmentation
SIGMA = 0.01       # ecart-type du bruit gaussien
MAX_SHIFT = 2      # decalage temporel maximum (frames)
N_COPIES = 2       # nombre de copies generees par sequence

rng = np.random.default_rng(42)

X_aug = [X_train]
y_aug = [y_train]

for copie in range(N_COPIES):
    X_copie = np.zeros_like(X_train)
    for i in range(N):
        seq = X_train[i].copy()

        # 1. Bruit gaussien sur les coordonnees
        bruit = rng.normal(0, SIGMA, size=seq.shape)
        seq = seq + bruit

        # 2. Jitter temporel (decalage circulaire)
        shift = rng.integers(-MAX_SHIFT, MAX_SHIFT + 1)
        if shift != 0:
            seq = np.roll(seq, shift, axis=0)

        X_copie[i] = seq

    X_aug.append(X_copie)
    y_aug.append(y_train)
    print(f"Copie {copie + 1} generee")

X_train_aug = np.concatenate(X_aug, axis=0)
y_train_aug = np.concatenate(y_aug, axis=0)

# Melanger le tout
indices = rng.permutation(len(X_train_aug))
X_train_aug = X_train_aug[indices]
y_train_aug = y_train_aug[indices]

print(f"\nX_train augmente : {X_train_aug.shape}")
print(f"y_train augmente : {y_train_aug.shape}")

np.save(dossier / "X_train_aug.npy", X_train_aug)
np.save(dossier / "y_train_aug.npy", y_train_aug)
print("\nFichiers sauvegardes : X_train_aug.npy, y_train_aug.npy")