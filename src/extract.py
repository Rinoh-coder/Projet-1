"""
Extraction des points du squelette sur le corpus complet.

Entree  : data/raw/{signeur}/{signe}/*.mp4
Sortie  : data/processed/X.npy     (N, 30, 225)
          data/processed/y.npy     (N,)
          data/processed/meta.csv  (fichier, signeur, signe, label)
          data/processed/classes.json (liste des signes dans l'ordre)
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def charger_config(chemin="config.yaml"):
    with open(chemin, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extraire_points_video(chemin_video, holistic):
    """Retourne un tableau (T, 225) des points corps + mains.

    Les points manquants sont remplaces par la derniere valeur connue
    (report), conformement a la pratique de la litterature sur les
    approches pose-based. Cela evite d'introduire un biais en
    remplissant avec des zeros, qui signifieraient "point a l'origine"
    plutot que "point absent".
    """
    cap = cv2.VideoCapture(str(chemin_video))
    frames = []
    n_sans_detection = 0

    derniere_pose = np.zeros((33, 3))
    derniere_main_g = np.zeros((21, 3))
    derniere_main_d = np.zeros((21, 3))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Corps : 33 points * 3 coords = 99 valeurs
        if results.pose_landmarks:
            corps = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.pose_landmarks.landmark])
            derniere_pose = corps
        else:
            corps = derniere_pose
            n_sans_detection += 1

        # Main gauche : 21 points * 3 coords = 63 valeurs
        if results.left_hand_landmarks:
            main_g = np.array([[lm.x, lm.y, lm.z]
                               for lm in results.left_hand_landmarks.landmark])
            derniere_main_g = main_g
        else:
            main_g = derniere_main_g

        # Main droite : 21 points * 3 coords = 63 valeurs
        if results.right_hand_landmarks:
            main_d = np.array([[lm.x, lm.y, lm.z]
                               for lm in results.right_hand_landmarks.landmark])
            derniere_main_d = main_d
        else:
            main_d = derniere_main_d

        vecteur = np.concatenate([corps.flatten(),
                                  main_g.flatten(),
                                  main_d.flatten()])  # (225,)
        frames.append(vecteur)

    cap.release()
    return np.array(frames), n_sans_detection

def reechantillonner(sequence, n_cible):
    """Ramene une sequence (T, D) a (n_cible, D) par interpolation."""
    T = len(sequence)
    if T == 0:
        return np.zeros((n_cible, sequence.shape[1] if len(sequence.shape) > 1 else 225))
    if T == n_cible:
        return sequence
    indices = np.linspace(0, T - 1, n_cible)
    return np.array([sequence[int(round(i))] for i in indices])


def normaliser(sequence):
    """Centre sur les epaules et divise par la distance entre epaules.

    Indices MediaPipe pose : epaule gauche = 11, epaule droite = 12.
    Chaque point est encode sur 3 valeurs, donc l'epaule gauche commence
    a l'indice 11*3 = 33.
    """
    seq = sequence.copy()
    for t in range(seq.shape[0]):
        epaule_g = seq[t, 11*3:11*3+3]
        epaule_d = seq[t, 12*3:12*3+3]
        centre = (epaule_g + epaule_d) / 2.0
        dist = np.linalg.norm(epaule_g - epaule_d)
        if dist < 1e-6:
            dist = 1.0
        # On normalise uniquement les points corps et mains (225 valeurs)
        for i in range(0, 225, 3):
            seq[t, i:i+3] = (seq[t, i:i+3] - centre) / dist
    return seq


def main():
    config = charger_config()
    signes = config["signes"]
    n_cible = config["extraction"]["nb_frames_cible"]
    seuil_echec = config["extraction"]["seuil_echec_detection"]

    classes = {signe: i for i, signe in enumerate(signes)}

    dossier_sortie = Path("data/processed")
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    # Recenser toutes les videos
    videos = []
    for chemin in sorted(Path("data/raw").rglob("*.mp4")):
        signeur = chemin.parent.parent.name
        signe = chemin.parent.name
        if signe not in classes:
            print(f"Signe inconnu, ignore : {chemin}")
            continue
        videos.append((chemin, signeur, signe))

    print(f"Videos a traiter : {len(videos)}")
    print(f"Frames cibles : {n_cible}")
    print(f"Seuil d'echec de detection : {seuil_echec*100:.0f}%\n")

    X = []
    y = []
    meta = []
    rejetees = []

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        for chemin, signeur, signe in tqdm(videos, desc="Extraction"):
            points, n_sans_detection = extraire_points_video(chemin, holistic)
            if len(points) == 0:
                rejetees.append((str(chemin), "video vide"))
                continue

            taux_echec = n_sans_detection / len(points)
            if taux_echec > seuil_echec:
                rejetees.append((str(chemin),
                                 f"echec detection {taux_echec*100:.1f}%"))
                continue

            seq = reechantillonner(points, n_cible)
            seq = normaliser(seq)

            X.append(seq)
            y.append(classes[signe])
            meta.append({
                "fichier": str(chemin),
                "signeur": signeur,
                "signe": signe,
                "label": classes[signe],
            })

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nSequences retenues : {len(X)}")
    print(f"Sequences rejetees : {len(rejetees)}")
    print(f"Forme X : {X.shape}")
    print(f"Forme y : {y.shape}")

    np.save(dossier_sortie / "X.npy", X)
    np.save(dossier_sortie / "y.npy", y)
    pd.DataFrame(meta).to_csv(dossier_sortie / "meta.csv", index=False)
    with open(dossier_sortie / "classes.json", "w", encoding="utf-8") as f:
        json.dump(signes, f, ensure_ascii=False, indent=2)

    if rejetees:
        with open(dossier_sortie / "rejetees.txt", "w", encoding="utf-8") as f:
            for chemin, raison in rejetees:
                f.write(f"{chemin}\t{raison}\n")
        print(f"\nDetail des rejets : data/processed/rejetees.txt")

    print(f"\nFichiers ecrits dans {dossier_sortie}/")


if __name__ == "__main__":
    main()