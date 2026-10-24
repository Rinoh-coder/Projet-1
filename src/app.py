"""
Application de demonstration : traduction de signes LSM vers texte et audio.
Moteur audio principal : MMS-TTS Malagasy (Facebook).
Fallback : pyttsx3 (voix systeme).
"""

import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Chargement du modele de classification
DOSSIER_MODELES = Path("models")
CHEMIN_MODELE = DOSSIER_MODELES / "model_gru.keras"
CHEMIN_CLASSES = Path("data/processed/classes.json")

with open(CHEMIN_CLASSES, encoding="utf-8") as f:
    CLASSES = json.load(f)

print("Chargement du modele de classification...")
MODELE = tf.keras.models.load_model(CHEMIN_MODELE)
print("Modele de classification charge.")

mp_holistic = mp.solutions.holistic

# --- Chargement du modele audio MMS-TTS Malagasy ---
print("Chargement du modele TTS malgache...")
try:
    from transformers import VitsModel, AutoTokenizer
    import torch

    TTS_MODEL = VitsModel.from_pretrained("facebook/mms-tts-mlg")
    TTS_TOKENIZER = AutoTokenizer.from_pretrained("facebook/mms-tts-mlg")
    MMS_DISPONIBLE = True
    print("Modele TTS malgache charge.")
except Exception as e:
    MMS_DISPONIBLE = False
    print(f"Modele TTS malgache indisponible ({e}). Fallback vers pyttsx3.")

# Fallback pyttsx3 
import pyttsx3

def generer_audio_pyttsx3(texte, chemin_sortie):
    """Fallback : genere un audio avec pyttsx3."""
    engine = pyttsx3.init()
    engine.save_to_file(texte, chemin_sortie)
    engine.runAndWait()
    return chemin_sortie

def generer_audio(texte, chemin_sortie):
    """Genere un fichier audio WAV pour le texte donne.
    
    Utilise MMS-TTS Malagasy si disponible, sinon pyttsx3.
    """
    if MMS_DISPONIBLE:
        try:
            inputs = TTS_TOKENIZER(texte, return_tensors="pt")
            with torch.no_grad():
                waveform = TTS_MODEL(**inputs).waveform

            import scipy.io.wavfile
            scipy.io.wavfile.write(
                chemin_sortie,
                rate=TTS_MODEL.config.sampling_rate,
                data=waveform.numpy().squeeze()
            )
            return chemin_sortie
        except Exception as e:
            print(f"Echec MMS-TTS ({e}). Fallback vers pyttsx3.")

    return generer_audio_pyttsx3(texte, chemin_sortie)

# Pretraitement des videos (identique a l'entrainement)
def extraire_points_video(chemin_video):
    cap = cv2.VideoCapture(str(chemin_video))
    frames = []
    derniere_pose = np.zeros((33, 3))
    derniere_main_g = np.zeros((21, 3))
    derniere_main_d = np.zeros((21, 3))

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            if results.pose_landmarks:
                corps = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                derniere_pose = corps
            else:
                corps = derniere_pose

            if results.left_hand_landmarks:
                main_g = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                derniere_main_g = main_g
            else:
                main_g = derniere_main_g

            if results.right_hand_landmarks:
                main_d = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                derniere_main_d = main_d
            else:
                main_d = derniere_main_d

            vecteur = np.concatenate([corps.flatten(), main_g.flatten(), main_d.flatten()])
            frames.append(vecteur)

    cap.release()
    return np.array(frames)

def reechantillonner(sequence, n_cible=30):
    T = len(sequence)
    if T == 0:
        return np.zeros((n_cible, 225))
    if T == n_cible:
        return sequence
    indices = np.linspace(0, T - 1, n_cible)
    return np.array([sequence[int(round(i))] for i in indices])

def normaliser(sequence):
    seq = sequence.copy()
    for t in range(seq.shape[0]):
        epaule_g = seq[t, 11*3:11*3+3]
        epaule_d = seq[t, 12*3:12*3+3]
        centre = (epaule_g + epaule_d) / 2.0
        dist = np.linalg.norm(epaule_g - epaule_d)
        if dist < 1e-6:
            dist = 1.0
        for i in range(0, 225, 3):
            seq[t, i:i+3] = (seq[t, i:i+3] - centre) / dist
    return seq

# Fonction principale 
def traduire_video(chemin_video):
    if chemin_video is None:
        return (
            gr.update(value="Statut : ERREUR - Aucune video fournie"),
            "Veuillez enregistrer ou uploader une video.",
            None
        )

    try:
        points = extraire_points_video(chemin_video)
        if len(points) == 0:
            return (
                gr.update(value="Statut : ERREUR - Video illisible"),
                "Impossible d'analyser la video.",
                None
            )

        seq = reechantillonner(points, 30)
        seq = normaliser(seq)
        seq = np.expand_dims(seq, axis=0)

        probas = MODELE.predict(seq, verbose=0)[0]
        idx = np.argmax(probas)
        confiance = probas[idx]
        signe_predit = CLASSES[idx]

        texte_resultat = f"Mot reconnu : {signe_predit}\nConfiance : {confiance*100:.1f}%"

        fichier_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        generer_audio(signe_predit, fichier_audio)

        return (
            gr.update(value=f"Statut : SUCCES - {signe_predit}"),
            texte_resultat,
            fichier_audio
        )

    except Exception as e:
        return (
            gr.update(value=f"Statut : ERREUR - {str(e)}"),
            "Une erreur est survenue lors du traitement.",
            None
        )

# Interface Gradio
theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="gray",
    neutral_hue="gray",
).set(
    body_background_fill="#f4f4f4",
    block_background_fill="#ffffff",
    block_border_width="2px",
    block_border_color="#d1d5db",
    button_primary_background_fill="#22c55e",
    button_primary_background_fill_hover="#16a34a",
)

with gr.Blocks(title="Traduction LSM", theme=theme) as demo:
    gr.Markdown(
        """
        # Traducteur de Langue des Signes Malgache (LSM)
        Prototype de reconnaissance de signes isoles avec sortie texte et audio.
        
        1. Utilisez l'onglet Webcam pour enregistrer un signe, ou Upload pour charger une video.
        2. Cliquez sur Traduire.
        3. Le resultat s'affiche avec la prononciation audio.
        """
    )

    statut = gr.Markdown("Statut : En attente d'une video...", visible=True)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Entree")
            entree_video = gr.Video(
                label="Enregistrer ou uploader le signe",
                sources=["upload", "webcam"],
                include_audio=False,
                height=300
            )
            bouton_traduire = gr.Button(
                "Traduire le signe",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("### Resultat")
            texte_sortie = gr.Textbox(
                label="Texte reconnu",
                lines=3,
                placeholder="Le resultat s'affichera ici...",
                interactive=False
            )
            audio_sortie = gr.Audio(
                label="Prononciation audio",
                type="filepath",
                interactive=False
            )

    bouton_traduire.click(
        fn=traduire_video,
        inputs=entree_video,
        outputs=[statut, texte_sortie, audio_sortie]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)