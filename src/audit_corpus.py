"""Audit read-only des données, des partitions et des transformations du corpus."""

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Fichier requis introuvable : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_metadata(classes, splits):
    with (PROCESSED / "meta.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 400:
        raise ValueError(f"400 séquences attendues, {len(rows)} trouvées.")
    if len({row["fichier"] for row in rows}) != len(rows):
        raise ValueError("Les chemins vidéo ne sont pas uniques dans meta.csv.")
    if any(row["signe"] not in classes for row in rows):
        raise ValueError("meta.csv contient une classe absente de classes.json.")

    signer_counts = Counter(row["signeur"] for row in rows)
    class_counts = Counter(row["signe"] for row in rows)

    print("Séquences par signeur :", dict(sorted(signer_counts.items())))
    print("Séquences par signe :", {name: class_counts[name] for name in classes})
    print("Distribution des partitions par classe :")

    memberships = {}
    for split_name in ("train", "val", "test"):
        indices = splits.get(split_name)
        if not isinstance(indices, list):
            raise ValueError(f"Partition invalide ou absente : {split_name}")
        if any(not isinstance(index, int) or index < 0 or index >= len(rows) for index in indices):
            raise ValueError(f"Indice invalide dans la partition {split_name}.")
        memberships[split_name] = set(indices)

    if set.union(*memberships.values()) != set(range(len(rows))):
        raise ValueError("Les partitions ne couvrent pas exactement les séquences.")
    if sum(len(indices) for indices in memberships.values()) != len(rows):
        raise ValueError("Les partitions se chevauchent.")

    train_signers = {rows[index]["signeur"] for index in memberships["train"]}
    val_signers = {rows[index]["signeur"] for index in memberships["val"]}
    test_signers = {rows[index]["signeur"] for index in memberships["test"]}
    if test_signers != {splits.get("signeur_test")}:
        raise ValueError("Le signeur de test ne correspond pas à splits.json.")
    if train_signers & test_signers or val_signers & test_signers:
        raise ValueError("Le signeur de test apparaît dans train ou validation.")

    for name in ("train", "val", "test"):
        counts = Counter(rows[index]["signe"] for index in memberships[name])
        expected = {"train": 21, "val": 9, "test": 10}[name]
        if any(counts[sign] != expected for sign in classes):
            raise ValueError(f"Distribution inattendue dans la partition {name}.")
        print(f"  {name}: {len(memberships[name])} séquences, "
              f"{expected} par classe")

    print("Signeurs de test :", sorted(test_signers))
    return rows


def audit_videos(rows):
    formats = Counter()
    durations = Counter()
    unreadable = []
    decode_mismatches = []

    for row in rows:
        path = ROOT / row["fichier"]
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,nb_read_frames,duration",
                "-of", "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if len(streams) != 1:
            unreadable.append(str(path))
            continue

        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        numerator, denominator = stream["r_frame_rate"].split("/")
        fps = round(int(numerator) / int(denominator), 3)
        reported_frames = int(stream["nb_frames"])
        decoded_frames = int(stream["nb_read_frames"])

        formats[(width, height, fps, reported_frames)] += 1
        durations[round(reported_frames / fps, 3)] += 1
        if decoded_frames != reported_frames:
            decode_mismatches.append(
                (str(path), reported_frames, decoded_frames)
            )

    if unreadable or decode_mismatches:
        raise ValueError(
            f"Vidéos illisibles : {len(unreadable)} ; "
            f"écarts de décodage : {len(decode_mismatches)}."
        )
    print("Formats vidéo (largeur, hauteur, fps, frames) :",
          dict(formats))
    print("Durées nominales (s) :", dict(durations))
    print("Échec d'ouverture/décodage : 0")


def audit_arrays(classes, splits):
    X = np.load(PROCESSED / "X.npy", mmap_mode="r")
    if X.shape != (400, 30, 225):
        raise ValueError(f"Forme inattendue pour X.npy : {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X.npy contient des valeurs non finies.")

    shoulder_left = X[:, :, 33:36]
    shoulder_right = X[:, :, 36:39]
    centers = (shoulder_left + shoulder_right) / 2
    widths = np.linalg.norm(shoulder_left - shoulder_right, axis=2)
    center_error = float(np.max(np.abs(centers)))
    width_error = float(np.max(np.abs(widths - 1)))
    print(f"Normalisation épaules, 12 000 frames : "
          f"erreur max centre={center_error:.2g}, "
          f"erreur max distance-unité={width_error:.2g}")

    y = np.load(PROCESSED / "y.npy", mmap_mode="r")
    if y.shape != (400,):
        raise ValueError(f"Forme inattendue pour y.npy : {y.shape}")
    with (PROCESSED / "meta.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_labels = np.array([classes.index(row["signe"]) for row in rows])
    if not np.array_equal(y, expected_labels):
        raise ValueError("Les étiquettes de y.npy ne correspondent pas à meta.csv.")

    X_train = np.load(PROCESSED / "X_train.npy", mmap_mode="r")
    if X_train.shape != (len(splits["train"]), 30, 225):
        raise ValueError(f"Forme inattendue pour X_train.npy : {X_train.shape}")
    ratios = []
    for sequence in X_train:
        adjacent = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
        endpoint = np.linalg.norm(sequence[0] - sequence[-1])
        ratios.append(float(endpoint / max(float(np.median(adjacent)), 1e-12)))
    ratios = np.asarray(ratios)
    print("Décalage circulaire, ratio distance(fin-début)/médiane(adjacent) : "
          f"médiane={np.median(ratios):.2f} ; "
          f"fin-début > médiane adjacente={int(np.sum(ratios > 1))}/"
          f"{len(ratios)}")


def main():
    classes = load_json(PROCESSED / "classes.json")
    splits = load_json(PROCESSED / "splits.json")
    rows = audit_metadata(classes, splits)
    audit_videos(rows)
    audit_arrays(classes, splits)


if __name__ == "__main__":
    main()
