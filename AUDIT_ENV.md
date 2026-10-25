# Audit de l'environnement d'exécution

Date de l'audit : 25 octobre 2026

## Structure vérifiée

Les éléments requis pour l'application sont présents : backend FastAPI, module
d'inférence, modèle `backend/models/model_gru.tflite`, frontend, classes et
données prétraitées. Le modèle n'a pas été modifié ; sa taille vérifiée est de
251 344 octets.

## Interpréteurs et environnements

- Le shell d'audit n'a pas de commande `python` dans son `PATH`.
- L'interpréteur système est `/usr/bin/python3`, version 3.12.3. Il ne contient
  pas les dépendances du backend.
- L'environnement d'exécution du projet est `backend/venv/bin/python`, version
  3.12.3. C'est cet interpréteur qui doit lancer Uvicorn.

## État initial des paquets du backend

| Paquet | Version observée | Résultat |
|---|---:|---|
| numpy | 1.26.4 | Conforme |
| mediapipe | 0.10.14 | Conforme |
| ai-edge-litert | 2.3.0 | Import et initialisation du modèle réussis |
| fastapi | 0.162.0 | Import réussi |
| uvicorn | 0.54.0 | Import réussi |
| opencv-contrib-python | 4.11.0.86 | Fournit `cv2`; import réussi |
| opencv-python-headless | absent de l'inventaire pip | Un résidu d'installation invalide était signalé |
| tflite-runtime | absent | Non requis par le backend |

`pip check` ne signale aucune dépendance cassée. `pip list` émet toutefois un
avertissement concernant un ancien résidu nommé
`~pencv-python-headless`. La version headless n'est pas reconnue comme un paquet
installé par pip. MediaPipe 0.10.14 déclare `opencv-contrib-python` comme
dépendance ; la configuration retient donc cette variante seule, plutôt que
d'installer deux distributions qui fournissent toutes deux le module `cv2`.
Le résidu n'a pas été supprimé pendant que le service existant était actif.

L'inventaire complet observé dans le venv backend :

```text
absl-py==2.5.1
ai-edge-litert==2.3.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
attrs==26.1.0
backports.strenum==1.2.8
certifi==2026.10.23
cffi==2.1.1
click==8.5.0
contourpy==1.3.3
cycler==0.12.1
fastapi==0.162.0
flatbuffers==25.12.19
fonttools==4.67.0
h11==0.16.0
idna==3.20
jax==0.4.38
jaxlib==0.4.38
kiwisolver==1.5.1
matplotlib==3.11.2
mediapipe==0.10.14
ml_dtypes==0.5.4
numpy==1.26.4
opencv-contrib-python==4.11.0.86
opentelemetry-api==1.45.1
opt_einsum==3.4.0
packaging==26.3
pillow==13.0.0
pip==24.0
protobuf==4.25.9
pycparser==3.12
pydantic==2.14.1
pydantic_core==2.50.1
pyparsing==3.3.3
python-dateutil==3.0.0
python-multipart==0.0.32
scipy==1.17.1
sentencepiece==0.2.2
six==1.17.0
sounddevice==0.5.6
starlette==1.10.0
tqdm==4.70.1
typing_extensions==4.16.0
typing-inspection==0.4.4
tzdata==2026.5
uvicorn==0.54.0
```

## Imports et code de démarrage

Les imports suivants passent dans le venv backend :

```text
numpy 1.26.4
mediapipe 0.10.14
ai-edge-litert OK
fastapi 0.162.0
uvicorn 0.54.0
opencv 4.11.0
```

`backend/inference.py` importe déjà `Interpreter` depuis
`ai_edge_litert.interpreter`. Aucun fichier du backend n'importe
`tflite_runtime`. Le backend n'importe pas TensorFlow. Des imports TensorFlow
existent encore dans l'ancien `src/app.py` et dans
`training/src/convert_tflite.py`; ils ne sont pas exécutés par le backend
d'inférence et n'ont pas été modifiés.

## Modifications de configuration

`backend/requirements.txt` fixe les versions critiques observées pendant
l'audit. NumPy est verrouillé à 1.26.4. `opencv-python-headless` a été retiré et
`opencv-contrib-python` conservé comme dépendance de MediaPipe. Les versions
FastAPI, Uvicorn, LiteRT et `python-multipart` sont également fixées à leurs
versions testées.

Aucun paquet n'a été réinstallé pendant cette vérification : le service existant
était déjà actif et répondait correctement. Le résidu pip
`~pencv-python-headless` reste à nettoyer lors d'une fenêtre où le venv pourra
être remis en état sans perturber un processus en cours. Le fichier de
dépendances permet désormais de recréer l'environnement sans demander la
distribution headless. Une résolution pip en mode `--dry-run` a confirmé que
toutes les versions verrouillées sont déjà satisfaites par le venv.

## Tests fonctionnels effectués

- Import de `app` et initialisation du modèle TFLite : réussis.
- Service local déjà actif : `/health`, `/` et `/docs` répondent HTTP 200.
- Après la mise à jour de la configuration, une nouvelle instance Uvicorn a été
  lancée sur le port 8001 puis arrêtée après les essais ; `/health`, `/` et
  `/docs` ont répondu HTTP 200.
- `POST /predire` avec `data/raw/signeur4/Miarahaba/01.mp4` : HTTP 200 avec un
  JSON contenant `signe` et `confiance`.
- Pour cette vidéo Miarahaba, le modèle a renvoyé `Solosaina` avec une confiance
  de 0,5456. L'endpoint et l'environnement fonctionnent ; cette prédiction
  incorrecte est une erreur de reconnaissance, pas une erreur de dépendance.
- Le chemin audio du navigateur n'a pas été retesté pendant cet audit Python.

## Commandes de lancement

Depuis la racine du dépôt :

```bash
source backend/venv/bin/activate
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
