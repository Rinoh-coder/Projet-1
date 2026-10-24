"""
Preparation des donnees avec un signeur reserve au test.

Regles :
  - Signeur4 : reserve au test, jamais vu a l'entrainement ni en validation.
  - Signeur1, signeur2, signeur3 : repartis en train (70%) et validation (30%),
    par groupe (signeur, signe) pour eviter toute fuite.
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


def main():
    config = charger_config()
    seed = config["seed"]
    rng = np.random.default_rng(seed)

    dossier = Path("data/processed")
    X = np.load(dossier / "X.npy")
    y = np.load(dossier / "y.npy")
    meta = pd.read_csv(dossier / "meta.csv")

    print(f"Corpus total : {len(X)} sequences")
    print(f"Graine utilisee : {seed}")
    print(f"Signeurs : {sorted(meta['signeur'].unique())}\n")

    SIGNEUR_TEST = "signeur4"
    if SIGNEUR_TEST not in meta["signeur"].values:
        print(f"Erreur : {SIGNEUR_TEST} absent du corpus.")
        return

    # Separer signeur4 (test) des autres
    masque_test = meta["signeur"] == SIGNEUR_TEST
    masque_trainval = ~masque_test

    indices_test = list(meta[masque_test].index)
    indices_trainval = list(meta[masque_trainval].index)

    print(f"Reserve pour le test ({SIGNEUR_TEST}) : {len(indices_test)} sequences")
    print(f"Disponible pour train/val : {len(indices_trainval)} sequences\n")

    # Grouper les indices train/val par (signeur, signe)
    groupes = defaultdict(list)
    for idx in indices_trainval:
        row = meta.loc[idx]
        cle = (row["signeur"], row["signe"])
        groupes[cle].append(idx)

    print(f"Groupes (signeur, signe) pour train/val : {len(groupes)}")

    indices_train = []
    indices_val = []

    for cle, indices in sorted(groupes.items()):
        n = len(indices)
        # 70% train, 30% val par groupe
        n_train = max(1, int(round(n * 0.7)))
        n_val = n - n_train

        indices_list = list(indices)
        rng.shuffle(indices_list)
        indices_train.extend(indices_list[:n_train])
        indices_val.extend(indices_list[n_train:])

    rng.shuffle(indices_train)
    rng.shuffle(indices_val)
    rng.shuffle(indices_test)

    X_train = X[indices_train]
    X_val = X[indices_val]
    X_test = X[indices_test]
    y_train = y[indices_train]
    y_val = y[indices_val]
    y_test = y[indices_test]

    print(f"\nTrain : {len(X_train)} sequences")
    print(f"Val   : {len(X_val)} sequences")
    print(f"Test  : {len(X_test)} sequences (signeur4 uniquement)")

    # Distribution par signe
    with open(dossier / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    print(f"\nDistribution par signe :")
    print(f"{'Signe':<15} {'Train':>6} {'Val':>6} {'Test':>6}")
    for i, signe in enumerate(classes):
        n_tr = int((y_train == i).sum())
        n_va = int((y_val == i).sum())
        n_te = int((y_test == i).sum())
        print(f"{signe:<15} {n_tr:>6} {n_va:>6} {n_te:>6}")

    # Verifications
    set_train = set(indices_train)
    set_val = set(indices_val)
    set_test = set(indices_test)
    assert set_train.isdisjoint(set_val), "Fuite train/val"
    assert set_train.isdisjoint(set_test), "Fuite train/test"
    assert set_val.isdisjoint(set_test), "Fuite val/test"
    assert len(set_train | set_val | set_test) == len(X), "Indices manquants"
    print("\nVerifications de non-fuite : OK")

    # Sauvegarde
    np.save(dossier / "X_train.npy", X_train)
    np.save(dossier / "X_val.npy", X_val)
    np.save(dossier / "X_test.npy", X_test)
    np.save(dossier / "y_train.npy", y_train)
    np.save(dossier / "y_val.npy", y_val)
    np.save(dossier / "y_test.npy", y_test)

    splits = {
        "seed": seed,
        "regle": "signeur4 reserve au test ; signeur1-3 repartis 70/30 train/val par groupe",
        "signeur_test": SIGNEUR_TEST,
        "train": [int(i) for i in indices_train],
        "val": [int(i) for i in indices_val],
        "test": [int(i) for i in indices_test],
    }
    with open(dossier / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"\nFichiers ecrits dans {dossier}/")


if __name__ == "__main__":
    main()