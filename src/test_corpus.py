import cv2
import mediapipe as mp
from pathlib import Path

mp_holistic = mp.solutions.holistic

dossiers = sorted(Path("data/raw/signeur2").glob("*"))
resultats = []

with mp_holistic.Holistic(static_image_mode=False) as holistic:
    for dossier in dossiers:
        videos = sorted(dossier.glob("*.mp4"))
        if not videos:
            continue
        video = videos[0]
        cap = cv2.VideoCapture(str(video))
        n_frames = 0
        n_pose = 0
        n_main = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            n_frames += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if results.pose_landmarks:
                n_pose += 1
            if results.left_hand_landmarks or results.right_hand_landmarks:
                n_main += 1
        cap.release()
        resultats.append((dossier.name, n_frames,
                          n_pose/n_frames*100, n_main/n_frames*100))

print(f"{'Signe':<15} {'Frames':>7} {'Pose%':>7} {'Main%':>7}")
for nom, n, p, m in resultats:
    print(f"{nom:<15} {n:>7} {p:>6.1f}% {m:>6.1f}%")