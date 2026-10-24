# Mission : Refonte complète du projet de traduction LSM

Tu es un ingénieur logiciel senior. Ta mission est de restructurer et refondre le projet existant de traduction de signes isolés de la Langue des Signes Malgache (LSM) vers texte et audio. Le projet actuel est fonctionnel mais monolithique (Gradio, TensorFlow complet, PyTorch). Tu dois le transformer en une architecture propre, légère, et défendable académiquement.

Travaille par étapes atomiques. Ne passe à l'étape suivante qu'après avoir vérifié que la précédente fonctionne. Documente chaque décision dans un fichier `LOG_REFONTE.md` à la racine du dépôt.

## Contexte du projet

- **Objectif scientifique** : comparer LSTM et GRU sur des points de squelette MediaPipe pour la reconnaissance de 10 signes isolés de LSM. Le modèle retenu est un GRU (137 610 paramètres, 90 % d'accuracy sur un signeur de test non vu à l'entraînement).
- **Données existantes** : `data/processed/` contient `X_train.npy`, `X_val.npy`, `X_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`, `classes.json`. Le modèle entraîné est dans `models/model_gru.keras`.
- **Contraintes matérielles** : la machine de démonstration est un ThinkPad avec 4 Go de RAM, sans GPU. L'objectif est de réduire drastiquement l'empreinte mémoire du backend.
- **Langue cible pour l'audio** : malgache. Un modèle ONNX existe : `elloza/mms-tts-mlg-onnx` [citation:4].

## Architecture cible

Séparer le projet en trois dossiers indépendants :
Projet-M1/
├── backend/
│ ├── app.py # FastAPI, endpoint POST /predire
│ ├── inference.py # TFLite interpreter, MediaPipe, pretraitement
│ ├── requirements.txt # fastapi, uvicorn, mediapipe, tflite-runtime, numpy, opencv-python-headless
│ └── models/
│ └── model_gru.tflite # converti depuis model_gru.keras
│
├── frontend/
│ ├── index.html # interface utilisateur
│ ├── app.js # fetch vers /predire + Transformers.js pour l'audio
│ └── style.css # design gris, voyants vert/jaune/rouge
│
├── training/ # code existant de capture, extraction, entrainement
│ ├── src/
│ ├── notebooks/
│ └── data/ # raw et processed (non versionne)
│
├── memoire/ # LaTeX (ne pas toucher)
├── SCOPE.md # existant, ne pas modifier sans accord
├── LOG.md # journal des decisions
└── README.md # mis a jour avec la nouvelle architecture


## Étape 1 — Conversion du modèle en TFLite

**Objectif** : produire `backend/models/model_gru.tflite` à partir de `models/model_gru.keras`.

**Instructions** :
1. Crée un script `training/src/convert_tflite.py` qui charge le modèle Keras et le convertit avec quantification dynamique (`optimizations=[tf.lite.Optimize.DEFAULT]`). Cela réduit la taille d'environ 4× sans perte significative de précision [citation:12].
2. Le script doit afficher la taille du fichier `.tflite` produit et vérifier que la conversion a réussi en chargeant le modèle avec `tf.lite.Interpreter`.
3. Sauvegarde le fichier dans `backend/models/model_gru.tflite`.
4. **Critère de vérification** : le fichier existe, sa taille est inférieure à 500 Ko, et le chargement avec `tf.lite.Interpreter` ne lève pas d'erreur.

**Note technique** : si la conversion échoue à cause d'une couche GRU stateful, vérifie que `stateful=False` dans la définition du modèle. Le modèle actuel n'est pas stateful, donc la conversion devrait passer.

## Étape 2 — Backend FastAPI avec TFLite

**Objectif** : écrire un backend léger qui reçoit une vidéo, extrait les points MediaPipe, exécute l'inférence TFLite, et renvoie un JSON.

**Instructions** :
1. Crée `backend/requirements.txt` avec : `fastapi`, `uvicorn`, `mediapipe`, `tflite-runtime`, `numpy`, `opencv-python-headless`.
2. Crée `backend/inference.py` :
   - Charge `model_gru.tflite` avec `tflite_runtime.interpreter.Interpreter`.
   - Réutilise **exactement** le pipeline de prétraitement de `training/src/extract.py` : extraction des 33 points du corps et 21 points par main, rééchantillonnage à 30 frames, normalisation par les épaules. Ne réinvente pas le prétraitement, copie-le pour garantir la cohérence avec l'entraînement.
   - Expose une fonction `predire(video_bytes) -> dict` qui renvoie `{"signe": str, "confiance": float}`.
3. Crée `backend/app.py` :
   - Initialise FastAPI.
   - Endpoint `POST /predire` qui accepte un fichier vidéo (`UploadFile`), appelle `inference.predire`, et renvoie le JSON.
   - Sert les fichiers statiques du dossier `frontend/` à la racine (`StaticFiles`).
   - Lance avec `uvicorn app:app --host 0.0.0.0 --port 8000`.
4. **Critère de vérification** : l'endpoint `/predire` répond correctement à un `curl` avec une vidéo de test (`data/raw/signeur4/Miarahaba/01.mp4`), et renvoie un JSON avec le signe attendu et une confiance > 0.5. L'interface Swagger (`/docs`) est accessible.

**Contrainte** : ne charge **aucune** bibliothèque audio côté Python. Pas de `gTTS`, pas de `pyttsx3`, pas de `torch`, pas de `transformers`.

## Étape 3 — Frontend HTML/JS avec audio ONNX

**Objectif** : écrire une page HTML qui upload une vidéo, appelle le backend, affiche le résultat, et génère l'audio malgache côté navigateur avec Transformers.js.

**Instructions** :
1. Crée `frontend/index.html` :
   - Design gris clair (`#f4f4f4`), blocs blancs, bordures grises (`#d1d5db`).
   - Bandeau de statut avec voyant (gris = en attente, jaune = en cours, vert = succès, rouge = erreur). **Pas d'emoji**.
   - Deux colonnes : entrée (upload vidéo + bouton "Traduire") et sortie (texte + lecteur audio).
2. Crée `frontend/app.js` :
   - Au clic sur "Traduire" : envoie la vidéo à `POST /predire` via `fetch` avec `FormData`.
   - Affiche le texte reçu et met à jour le voyant.
   - Charge Transformers.js via CDN (`import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers'`).
   - Utilise le pipeline text-to-speech avec le modèle `elloza/mms-tts-mlg` [citation:4] :
     ```javascript
     const synthesizer = await pipeline('text-to-speech', 'elloza/mms-tts-mlg', { quantized: false });
     const output = await synthesizer(texte);
     // output.audio est un Float32Array, output.sampling_rate = 16000

     - Convertis le `Float32Array` en fichier WAV jouable dans le navigateur, et affecte-le au lecteur audio.
     
     1. **Critère de vérification** : la page s'ouvre, l'upload d'une vidéo de test affiche le signe prédit, et le lecteur audio joue la prononciation malgache générée localement (pas de requête serveur pour l'audio).
     
     ## Étape 4 — Nettoyage de l'environnement Python
     **Objectif** : alléger l'environnement local en supprimant les dépendances lourdes devenues inutiles.
     
     **Instructions** :
     
     1. Dans `backend/requirements.txt`, ne garder que les dépendances listées à l'étape 2.
     2. Documente dans `README.md` les commandes pour recréer un venv propre pour le backend :

     cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app

1. ```

```
2. Vérifie que `python -c "import tensorflow"` échoue (ce qui est normal, TensorFlow n'est plus nécessaire pour le backend) mais que `python -c "import tflite_runtime.interpreter"` fonctionne.

## Étape 5 — Conventions Git et workflow
**Conventions de commit** (format Conventional Commits) :

- `feat(backend): ajout endpoint /predire avec TFLite`
- `feat(frontend): ajout upload video et appel API`
- `feat(frontend): integration Transformers.js pour audio malgache`
- `refactor(backend): suppression dependances TensorFlow completes`
- `docs(readme): mise a jour architecture et instructions`
- `fix(frontend): correction chargement modele ONNX`
**Workflow Git** :

1. Crée une branche `refonte/architecture-legere` depuis `main`.
2. Travaille par commits atomiques. Chaque commit doit correspondre à une étape fonctionnelle.
3. Ne pushe jamais directement sur `main`.
4. À la fin, ouvre une Pull Request vers `main` avec une description listant :

- Ce qui a changé (architecture, dépendances, code).
- Ce qui a été testé (endpoint, audio, interface).
- Ce qui reste à faire (webcam temps réel, tests de robustesse).
5. Ne merge pas toi-même. L'utilisateur merge après relecture du diff.

## Règles générales

1. **Ne modifie pas** les fichiers du dossier `training/` (le code d'entraînement existant doit rester intact pour référence).
2. **Ne modifie pas** `SCOPE.md`, `LOG.md`, ou `memoire/`.
3. **Documente chaque décision** dans `LOG_REFONTE.md` : pourquoi TFLite, pourquoi Transformers.js, pourquoi FastAPI, quel gain de ressources attendu.
4. **Pas d'emoji** dans le code, les commentaires, l'interface, ou les logs.
5. **Pas de dépendance non justifiée**. Chaque bibliothèque ajoutée doit être justifiée par un besoin fonctionnel précis.
6. Si une étape échoue, **ne passe pas à la suivante**. Documente l'échec dans `LOG_REFONTE.md` et propose une alternative.
7. À la fin de chaque étape, affiche un résumé de ce qui a été fait et vérifie que le critère de vérification est rempli.

## Livrables finaux attendus

- `backend/` fonctionnel avec FastAPI + TFLite + MediaPipe.
- `frontend/` fonctionnel avec upload vidéo + audio ONNX malgache.
- `backend/models/model_gru.tflite` présent et fonctionnel.
- `README.md` mis à jour avec les instructions de lancement.
- `LOG_REFONTE.md` complet avec les décisions et les gains mesurés.
- Une branche `refonte/architecture-legere` avec des commits atomiques et une PR prête pour relecture.
Commence par l'étape 1. Affiche le plan d'action pour cette étape avant de créer le script.

