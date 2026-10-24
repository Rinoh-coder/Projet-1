# Projet-1

## Lancer l'application en local

Les commandes ci-dessous sont à exécuter dans un terminal Linux/macOS. Commencez
dans le dossier racine du dépôt `Projet-M1`, c'est-à-dire le dossier qui contient
`README.md`, `backend/` et `frontend/`. Si le terminal n'y est pas déjà, placez-vous
dans ce dossier avec `cd /chemin/vers/Projet-M1`.

### 1. Créer l'environnement et installer les dépendances

À exécuter depuis la racine `Projet-M1` :

```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

L'activation réussie de l'environnement affiche normalement `(venv)` au début
de l'invite de commande. Gardez ce terminal ouvert pour lancer l'application.

### 2. Vérifier le modèle puis démarrer le serveur

Toujours dans le même terminal, depuis la racine `Projet-M1` :

```bash
test -f backend/models/model_gru.tflite && echo "Modèle TFLite présent"
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Ne fermez pas ce terminal : le serveur doit rester lancé pendant l'utilisation.
FastAPI charge le modèle TFLite au démarrage et sert à la fois l'API et le
frontend. Il n'est pas nécessaire de démarrer un serveur frontend séparé.

### 3. Ouvrir et utiliser l'application

Dans un navigateur, ouvrez exactement :

```text
http://127.0.0.1:8000
```

Autorisez l'accès à la caméra. Cliquez sur **Capturer**, faites le signe après
le compte à rebours, puis attendez le résultat et la synthèse vocale. La caméra
est utilisée depuis `localhost`, qui est autorisé par les navigateurs.

La première synthèse vocale télécharge le modèle ONNX malgache dans le
navigateur ; une connexion Internet est nécessaire à ce premier chargement.

### Vérifier que le serveur fonctionne

Laissez le serveur tourner et, dans un second terminal, exécutez depuis la
racine `Projet-M1` :

```bash
curl http://127.0.0.1:8000/health
```

La réponse attendue est `{"status":"ok","modele":"model_gru.tflite"}`. La
documentation de l'API est accessible sur `http://127.0.0.1:8000/docs`.

### En cas d'erreur « Failed to fetch »

- Vérifiez que le terminal du serveur affiche qu'il est en cours d'exécution.
- Vérifiez l'URL : l'application doit être ouverte sur le port **8000**,
  jamais sur un serveur statique comme `http://127.0.0.1:8765`.
- Si le port 8000 est déjà utilisé, arrêtez l'autre serveur avec `Ctrl+C`, puis
  relancez Uvicorn.
- Arrêtez le serveur avec `Ctrl+C`. Pour quitter l'environnement virtuel,
  exécutez `deactivate`.
