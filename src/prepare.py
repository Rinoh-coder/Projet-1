"""
Preparation des donnees : separation train / validation / test.

La separation est groupee par (signeur, signe) pour eviter toute fuite :
aucune sequence d'un meme groupe ne se retrouve dans deux ensembles.

Ratio par groupe : 7 train / 2 val / 1 test.

Entree  : data/processed/X.npy, y.npy, meta.csv
Sortie  : data/processed/X_train.npy, X_val.npy, X_test.npy
          data/processed/y_train.npy, y_val.npy, y_test.npy
          data/processed/splits.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def charger_config(chemin="config.yaml"):
    with open(chemin, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repartir_groupe(indices, rng, n_train=7, n_val=2, n_test=1):
    """Repartit les indices d'un groupe en train/val/test."""
    indices = list(indices)
    rng.shuffle(indices)
    train = indices[:n_train]
    val = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def main():
    config = charger_config()
    seed = config["seed"]
    rng = np.random.default_rng(seed)

    dossier = Path("data/processed")
    X = np.load(dossier / "X.npy")
    y = np.load(dossier / "y.npy")
    meta = pd.read_csv(dossier / "meta.csv")

    print(f"Corpus total : {len(X)} sequences")
    print(f"Graine utilisee : {seed}\n")

    # Regrouper les indices par (signeur, signe)
    groupes = defaultdict(list)
    for idx, row in meta.iterrows():
        cle = (row["signeur"], row["signe"])
        groupes[cle].append(idx)

    print(f"Nombre de groupes (signeur, signe) : {len(groupes)}")
    tailles = [len(v) for v in groupes.values()]
    print(f"Taille par groupe : min={min(tailles)}, max={max(tailles)}, "
          f"moyenne={np.mean(tailles):.1f}\n")

    indices_train = []
    indices_val = []
    indices_test = []

    for cle, indices in sorted(groupes.items()):
        n = len(indices)
        if n < 10:
            print(f"Attention : groupe {cle} n'a que {n} sequences, "
                  f"repartition ajustee.")
            n_train = max(1, int(round(n * 0.7)))
            n_val = max(1, int(round(n * 0.2)))
            n_test = max(0, n - n_train - n_val)
        else:
            n_train, n_val, n_test = 7, 2, 1

        train, val, test = repartir_groupe(indices, rng,
                                           n_train, n_val, n_test)
        indices_train.extend(train)
        indices_val.extend(val)
        indices_test.extend(test)

    # Melanger l'ordre dans chaque ensemble
    rng.shuffle(indices_train)
    rng.shuffle(indices_val)
    rng.shuffle(indices_test)

    X_train = X[indices_train]
    X_val = X[indices_val]
    X_test = X[indices_test]
    y_train = y[indices_train]
    y_val = y[indices_val]
    y_test = y[indices_test]

    print(f"Train : {len(X_train)} sequences")
    print(f"Val   : {len(X_val)} sequences")
    print(f"Test  : {len(X_test)} sequences")
    print(f"Total : {len(X_train) + len(X_val) + len(X_test)}\n")

    # Verification : chaque signe doit etre represente dans chaque ensemble
    print("Distribution par signe dans chaque ensemble :")
    with open(dossier / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    print(f"{'Signe':<15} {'Train':>6} {'Val':>6} {'Test':>6}")
    for i, signe in enumerate(classes):
        n_tr = int((y_train == i).sum())
        n_va = int((y_val == i).sum())
        n_te = int((y_test == i).sum())
        print(f"{signe:<15} {n_tr:>6} {n_va:>6} {n_te:>6}")

    # Sauvegarde
    np.save(dossier / "X_train.npy", X_train)
    np.save(dossier / "X_val.npy", X_val)
    np.save(dossier / "X_test.npy", X_test)
    np.save(dossier / "y_train.npy", y_train)
    np.save(dossier / "y_val.npy", y_val)
    np.save(dossier / "y_test.npy", y_test)

    splits = {
        "seed": seed,
        "ratio": "7/2/1 par groupe (signeur, signe)",
        "train": [int(i) for i in indices_train],
        "val": [int(i) for i in indices_val],
        "test": [int(i) for i in indices_test],
    }
    with open(dossier / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\nFichiers ecrits dans {dossier}/")


if __name__ == "__main__":
    main()