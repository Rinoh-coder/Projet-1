"""MediaPipe preprocessing and TFLite inference for isolated LSM signs."""

import json
import tempfile
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter


BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "models" / "model_gru.tflite"
CLASSES_PATH = BACKEND_DIR / "classes.json"
TARGET_FRAMES = 30
MAX_NO_POSE_RATIO = 0.20

with CLASSES_PATH.open(encoding="utf-8") as classes_file:
    CLASSES = json.load(classes_file)

if not isinstance(CLASSES, list) or not all(
    isinstance(label, str) for label in CLASSES
):
    raise ValueError(f"Invalid class list in {CLASSES_PATH}")

INTERPRETER = Interpreter(model_path=str(MODEL_PATH), num_threads=1)
INTERPRETER.allocate_tensors()
INPUT_DETAILS = INTERPRETER.get_input_details()[0]
OUTPUT_DETAILS = INTERPRETER.get_output_details()[0]

if int(OUTPUT_DETAILS["shape"][-1]) != len(CLASSES):
    raise ValueError(
        f"Model has {OUTPUT_DETAILS['shape'][-1]} outputs but "
        f"{len(CLASSES)} class names are configured."
    )

_INTERPRETER_LOCK = Lock()


class InvalidVideoError(ValueError):
    """Raised when an uploaded video cannot be processed as a sign sample."""


def extraire_points_video(chemin_video, holistic):
    """Return frame landmarks and count frames without pose detection."""
    cap = cv2.VideoCapture(str(chemin_video))
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoError("La video ne peut pas etre ouverte.")

    frames = []
    n_sans_detection = 0
    derniere_pose = np.zeros((33, 3))
    derniere_main_g = np.zeros((21, 3))
    derniere_main_d = np.zeros((21, 3))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            if results.pose_landmarks:
                corps = np.array(
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                )
                derniere_pose = corps
            else:
                corps = derniere_pose
                n_sans_detection += 1

            if results.left_hand_landmarks:
                main_g = np.array(
                    [
                        [lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark
                    ]
                )
                derniere_main_g = main_g
            else:
                main_g = derniere_main_g

            if results.right_hand_landmarks:
                main_d = np.array(
                    [
                        [lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark
                    ]
                )
                derniere_main_d = main_d
            else:
                main_d = derniere_main_d

            vecteur = np.concatenate(
                [corps.flatten(), main_g.flatten(), main_d.flatten()]
            )
            frames.append(vecteur)
    finally:
        cap.release()

    return np.array(frames), n_sans_detection


def reechantillonner(sequence, n_cible):
    """Resample a sequence to n_cible frames using the training pipeline."""
    frame_count = len(sequence)
    if frame_count == 0:
        feature_count = sequence.shape[1] if len(sequence.shape) > 1 else 225
        return np.zeros((n_cible, feature_count))
    if frame_count == n_cible:
        return sequence
    indices = np.linspace(0, frame_count - 1, n_cible)
    return np.array([sequence[int(round(index))] for index in indices])


def normaliser(sequence):
    """Normalize body and hand landmarks around the shoulder midpoint."""
    seq = sequence.copy()
    for frame_index in range(seq.shape[0]):
        epaule_g = seq[frame_index, 11 * 3 : 11 * 3 + 3]
        epaule_d = seq[frame_index, 12 * 3 : 12 * 3 + 3]
        centre = (epaule_g + epaule_d) / 2.0
        dist = np.linalg.norm(epaule_g - epaule_d)
        if dist < 1e-6:
            dist = 1.0
        for point_index in range(0, 225, 3):
            seq[frame_index, point_index : point_index + 3] = (
                seq[frame_index, point_index : point_index + 3] - centre
            ) / dist
    return seq


def predire(video_bytes: bytes) -> dict:
    """Predict an isolated sign from uploaded video bytes."""
    if not video_bytes:
        raise InvalidVideoError("Le fichier video est vide.")

    suffix = ".webm" if video_bytes.startswith(b"\x1a\x45\xdf\xa3") else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as video_file:
        video_file.write(video_bytes)
        chemin_video = Path(video_file.name)

    try:
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(static_image_mode=False) as holistic:
            points, n_sans_detection = extraire_points_video(
                chemin_video, holistic
            )

        if len(points) == 0:
            raise InvalidVideoError("Aucune frame exploitable dans la video.")
        taux_echec = n_sans_detection / len(points)
        if taux_echec > MAX_NO_POSE_RATIO:
            raise InvalidVideoError(
                "La pose est detectee dans moins de 80 % des frames."
            )

        sequence = reechantillonner(points, TARGET_FRAMES)
        sequence = normaliser(sequence).astype(np.float32)
        batch = np.expand_dims(sequence, axis=0)

        with _INTERPRETER_LOCK:
            INTERPRETER.set_tensor(INPUT_DETAILS["index"], batch)
            INTERPRETER.invoke()
            probabilites = INTERPRETER.get_tensor(OUTPUT_DETAILS["index"])[0]

        index = int(np.argmax(probabilites))
        return {
            "signe": CLASSES[index],
            "confiance": float(probabilites[index]),
        }
    finally:
        chemin_video.unlink(missing_ok=True)
