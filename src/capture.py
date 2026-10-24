"""
Script de capture des signes LSM avec apercu des landmarks MediaPipe.

Usage :
    python src/capture.py --signeur signeur1
    python src/capture.py --signeur signeur1 --signe Miarahaba
    python src/capture.py --signeur signeur1 --repetitions 12
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import yaml


def charger_config(chemin="config.yaml"):
    with open(chemin, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dessiner_landmarks(image, results, mp_holistic, mp_drawing):
    """Dessine les landmarks sur une copie de l'image (pour l'affichage)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 200, 0), thickness=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 0, 0), thickness=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 0, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 0, 200), thickness=2))
    return image


def statut_detection(results):
    """Retourne un texte court indiquant ce qui est detecte."""
    parties = []
    if results.pose_landmarks:
        parties.append("corps")
    if results.left_hand_landmarks:
        parties.append("main G")
    if results.right_hand_landmarks:
        parties.append("main D")
    return " + ".join(parties) if parties else "aucune detection"


def compte_a_rebours(cap, signe, repetition, total_reps, duree,
                     holistic, mp_holistic, mp_drawing):
    """Affiche un compte a rebours avec apercu des landmarks."""
    for i in range(duree, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = dessiner_landmarks(image, results, mp_holistic, mp_drawing)

        cv2.putText(image, f"Signe : {signe}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(image, f"Repetition {repetition}/{total_reps}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Prepare-toi : {i}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)
        cv2.putText(image, statut_detection(results), (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Capture LSM", image)
        cv2.waitKey(1000)


def enregistrer_sequence(cap, chemin, duree_sec, fps, signe, repetition, total,
                         holistic, mp_holistic, mp_drawing):
    """Enregistre une sequence et affiche les landmarks en direct."""
    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(chemin), fourcc, fps, (largeur, hauteur))

    nb_frames_total = int(duree_sec * fps)
    frame_count = 0

    while frame_count < nb_frames_total:
        ret, frame = cap.read()
        if not ret:
            break

        # On ecrit la frame ORIGINALE dans le fichier (sans landmarks)
        writer.write(frame)

        # On traite une copie pour l'affichage
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = dessiner_landmarks(image, results, mp_holistic, mp_drawing)

        cv2.putText(image, f"ENREGISTREMENT : {signe}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(image, f"Repetition {repetition}/{total}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Frame {frame_count}/{nb_frames_total}",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, statut_detection(results), (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Capture LSM", image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Capture des signes LSM")
    parser.add_argument("--signeur", required=True,
                        help="Nom du signeur (ex: signeur1)")
    parser.add_argument("--signe", default=None,
                        help="Capturer un seul signe (sinon tous)")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="Nombre de repetitions (sinon config)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = charger_config(args.config)
    signes = config["signes"]
    if args.signe:
        if args.signe not in signes:
            print(f"Erreur : '{args.signe}' n'est pas dans la liste des signes.")
            print(f"Signes valides : {signes}")
            return
        signes = [args.signe]

    nb_reps = args.repetitions or config["capture"]["nb_repetitions_par_signeur"]
    duree_sec = config["capture"]["duree_secondes"]
    duree_rebours = config["capture"]["compte_a_rebours"]
    fps = config["capture"]["fps"]
    largeur = config["capture"]["resolution_largeur"]
    hauteur = config["capture"]["resolution_hauteur"]

    dossier_base = Path("data/raw") / args.signeur
    dossier_base.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la camera sur /dev/video0.")
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    print(f"\n=== Capture pour {args.signeur} ===")
    print(f"Signes : {len(signes)}")
    print(f"Repetitions par signe : {nb_reps}")
    print(f"Duree par sequence : {duree_sec}s")
    print(f"Appuie sur 'q' pour interrompre.\n")

    total_capturees = 0
    total_prevues = len(signes) * nb_reps

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        for signe in signes:
            dossier_signe = dossier_base / signe
            dossier_signe.mkdir(parents=True, exist_ok=True)

            existantes = sorted(dossier_signe.glob("*.mp4"))
            nb_existantes = len(existantes)

            if nb_existantes >= nb_reps:
                print(f"[{signe}] deja complet ({nb_existantes}/{nb_reps}), ignore.")
                total_capturees += nb_existantes
                continue

            print(f"\n[{signe}] {nb_existantes}/{nb_reps} deja faites, "
                  f"reprise a la repetition {nb_existantes + 1}.")

            for rep in range(nb_existantes + 1, nb_reps + 1):
                ret, frame = cap.read()
                if not ret:
                    print("Erreur de lecture camera.")
                    break

                compte_a_rebours(cap, signe, rep, nb_reps, duree_rebours,
                                 holistic, mp_holistic, mp_drawing)

                chemin = dossier_signe / f"{rep:02d}.mp4"
                n = enregistrer_sequence(cap, chemin, duree_sec, fps,
                                         signe, rep, nb_reps,
                                         holistic, mp_holistic, mp_drawing)
                total_capturees += 1
                print(f"  [{signe}] repetition {rep}/{nb_reps} -> "
                      f"{n} frames sauvegardees")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n=== Termine ===")
    print(f"Sequences capturees : {total_capturees}/{total_prevues}")
    print(f"Dossier : {dossier_base}")


if __name__ == "__main__":
    main()
