# Journal de refonte

## Étape 1 — Conversion du modèle en TFLite

### Décisions

- Un nouveau script `training/src/convert_tflite.py` est prévu pour convertir
  `models/model_gru.keras` avec `tf.lite.Optimize.DEFAULT`. Le code
  d'entraînement déjà présent à la racine (`src/`) n'est pas modifié.
- Le modèle GRU est non stateful, mais sa configuration `unroll=False` a
  empêché l'export en opérateurs TFLite intégrés (`TensorListReserve`).
  L'export clone donc l'architecture en activant le déroulage uniquement sur
  les couches GRU, avec le nombre de frames statique déjà attendu par le modèle
  (30), puis conserve exactement les poids entraînés. Cela évite d'exiger les
  opérateurs Select TF/Flex et TensorFlow complet dans le backend.
- La conversion vérifie la taille cible de moins de 500 KiB et charge le modèle
  converti avec `tf.lite.Interpreter` avant d'écrire le fichier. Cette
  vérification évite de publier un artefact incomplet ou qui ne respecte pas le
  critère de taille.
- TFLite est retenu pour limiter l'empreinte mémoire et la taille du runtime
  d'inférence sur la machine de démonstration.

### Vérification

- Conversion exécutée avec TensorFlow CPU 2.19.1 dans un environnement
  temporaire isolé. Le venv préexistant est incohérent (SciPy 1.18.1 avec
  NumPy 1.26.4 et TensorFlow CPU 2.16.2 avec un binaire lié à l'API NumPy 2) ;
  aucune dépendance de conversion n'a été ajoutée aux manifestes du projet.
- Le fichier `backend/models/model_gru.tflite` a une taille de 251 344 octets
  (245,5 KiB), inférieure au seuil de 500 KiB.
- `tf.lite.Interpreter` a chargé le fichier et alloué ses tenseurs sans erreur.
- Sur les 100 exemples de `X_test.npy` : accuracy Keras 90 %, accuracy TFLite
  90 %, accord des classes entre les deux modèles 100 %, erreur absolue maximale
  entre les sorties 0,010011.
- L'étape 1 est validée. Le fichier source de 1 677 751 octets a été réduit
  à 251 344 octets (réduction de taille de 85,0 %) pour l'artefact TFLite.

## Étape 2 — Backend FastAPI avec TFLite

### Décisions et implémentation en cours

- Le backend sépare l'API (`backend/app.py`) du traitement vidéo et de
  l'inférence (`backend/inference.py`), charge un interpréteur TFLite unique et
  sérialise son accès pour éviter les appels concurrents sur le même
  interpréteur.
- Le prétraitement réutilise les mêmes 225 coordonnées, la conservation des
  dernières détections, le rééchantillonnage à 30 frames et la normalisation
  par les épaules de `src/extract.py`. Les vidéos avec plus de 20 % de frames
  sans pose détectée sont rejetées comme dans l'extraction d'entraînement.
- `backend/classes.json` embarque l'ordre des classes afin que l'inférence ne
  dépende pas du dossier de données ignoré. `python-multipart` est ajouté car
  FastAPI en a besoin pour recevoir un upload multipart.
- La liste des signes et le modèle ont 10 sorties/classes, alors que
  `config.yaml` indiquait encore `nb_classes: 12`. Cette valeur est alignée sur
  les 10 signes sans modifier les autres changements préexistants du fichier.
- `.gitignore` autorise uniquement le modèle TFLite de `backend/models/`, sans
  cesser d'ignorer les autres modèles d'entraînement.

### Vérification et blocage

- Les modules backend passent l'analyse syntaxique Python et l'ordre des
  classes correspond au vocabulaire configuré. Le modèle TFLite est présent et
  non ignoré par Git.
- La machine disponible ne fournit que Python 3.12.3. `pip index versions
  tflite-runtime` ne trouve aucune distribution installable pour cet
  environnement. Sur choix utilisateur, le backend utilise donc
  `ai-edge-litert`, le runtime LiteRT compatible avec Python 3.12, au lieu de
  `tflite-runtime`. Le modèle reste un FlatBuffer TFLite et l'inférence reste
  locale ; cette adaptation est limitée au runtime et à son import Python.
- MediaPipe 1.1.1 (version sans borne proposée initialement par pip) n'expose
  plus `mp.solutions`, utilisée par le pipeline d'entraînement. La dépendance
  est bornée à 0.10.14, version installée dans le venv d'entraînement, avec
  NumPy < 2 et OpenCV 4.11 pour préserver les APIs binaires utilisées lors de
  l'extraction. OpenCV contrib est une dépendance transitive requise par
  MediaPipe ; ses versions et celles de JAX/JAXlib sont bornées pour stabiliser
  la résolution des dépendances, JAX étant requis par les métadonnées MediaPipe
  0.10 mais non importé par le backend.
- La première prédiction HTTP/manuelle sur la vidéo de référence était
  `Solosaina` à 0,545 alors que la vérité terrain est `Miarahaba`. Cette vidéo
  est dans `X_test` et le modèle TFLite la classe également mal depuis le
  vecteur sauvegardé. L'extraction backend différait du vecteur sauvegardé ;
  l'alignement sur MediaPipe 0.10.14 n'a pas supprimé cet écart (différence
  absolue maximale 1,32224), ce qui laisse à investiguer la reproductibilité de
  l'extraction historique.
- Vérification HTTP : `/docs` répond 200. `POST /predire` accepte le fichier et
  renvoie `{"signe":"Solosaina","confiance":0.5455507040023804}` ; la confiance
  dépasse 0,5, mais le signe ne correspond pas à la vérité terrain.
- Le vecteur correspondant dans `X_test` est lui aussi classé `Solosaina`
  (confiance 0,55085) alors que son étiquette est `Miarahaba`. Le modèle TFLite
  obtient 90 % sur l'ensemble de test, mais ce cas précis fait partie des
  erreurs du modèle. Le backend ne peut pas corriger cela sans modifier ou
  réentraîner le modèle, ce qui dépasserait la mission de migration.
- L'étape 2 n'est donc pas validée selon le critère d'acceptation demandé.
  Alternative proposée : fournir un modèle GRU réentraîné puis vérifier sur
  le jeu de test que cette vidéo est correctement classée, ou confirmer
  explicitement qu'un endpoint fonctionnel avec cette erreur connue est
  acceptable. L'étape 3 n'est pas commencée.
- L'utilisateur a choisi de fournir un modèle GRU corrigé et de valider ce cas
  avant de poursuivre. En attente de ce modèle à `models/model_gru.keras`.

## Étape 3 — Frontend HTML/JS avec audio ONNX

### Décisions et implémentation

- L'utilisateur a explicitement demandé de continuer le frontend malgré le
  critère de classification non satisfait à l'étape 2. Cette décision permet
  d'avancer sur l'interface sans présenter la prédiction connue comme correcte.
- `frontend/index.html` et `frontend/style.css` fournissent une interface
  responsive en deux cartes, un statut accessible, une prévisualisation vidéo,
  un résultat avec confiance, et un lecteur audio.
- `frontend/app.js` envoie le fichier via `FormData` à `/predire`, valide la
  réponse JSON et charge paresseusement le modèle de synthèse dans le navigateur.
  Le modèle ONNX est référencé par son identifiant public effectif
  `elloza/mms-tts-mlg-onnx` ; la variante quantifiée est utilisée pour limiter
  le transfert et la mémoire (fichier ONNX quantifié annoncé à environ 38 Mo).
  L'identifiant sans suffixe `elloza/mms-tts-mlg` n'est pas exposé comme modèle
  public autonome. La synthèse reste client-side : le signal Float32 est
  converti en WAV PCM puis fourni au lecteur par une URL `blob:`, sans requête
  audio au backend.
- Un seul thread WASM est configuré afin de limiter l'utilisation CPU/mémoire
  du navigateur sur la machine de démonstration. Les fichiers vidéo et audio
  temporaires du navigateur sont libérés lors de leur remplacement ou fermeture.

### Vérification

- `node --input-type=module --check` valide la syntaxe JavaScript ; le HTML se
  parse correctement et ses 15 identifiants sont uniques ; `git diff --check`
  ne relève pas d'erreur d'espacement.
- Le serveur FastAPI sert `/`, `/style.css` et `/app.js` avec HTTP 200, et
  `/docs` reste accessible.
- Test navigateur réel sur
  `data/raw/signeur4/Miarahaba/01.mp4` : réponse `Solosaina`, confiance 54,6 %
  (erreur du modèle déjà relevée), synthèse locale terminée, WAV d'environ
  1,424 seconde prêt et lisible depuis une URL `blob:` locale.
- L'étape frontend est validée fonctionnellement ; l'étape backend reste non
  validée scientifiquement pour cette vidéo. Le README global sera complété
  lors de la consolidation de l'architecture.

### Ajustement du parcours de capture

- L'interface est limitée à la caméra, au bouton Capturer, au compte à rebours
  de préparation, au résultat texte et au lecteur audio.
- Après le compte à rebours, une vidéo de deux secondes est enregistrée puis
  envoyée à `/predire`. Le signe prédit est vocalisé dans le navigateur par le
  modèle ONNX malgache ; aucun audio n'est envoyé au backend.
- Vérification : `node --check frontend/app.js`, analyse HTML et `git diff
  --check` réussis ; la page statique affiche la caméra et le bouton, avec le
  résultat et le lecteur audio masqués jusqu'à la première prédiction.
- Le `Failed to fetch` était causé par l'absence de serveur FastAPI actif :
  l'interface utilisait une URL relative vers `/predire`, mais le backend
  n'était pas lancé sur le port 8000. Un contrôle `/health` confirme désormais
  avant la capture que le backend et le modèle TFLite sont prêts, avec une
  indication explicite si l'interface est ouverte depuis un serveur statique.
- La capture vise 60 images et ajuste sa durée à la fréquence réelle de la
  caméra (environ deux secondes à 30 images/s). Le backend identifie maintenant
  le conteneur WebM de `MediaRecorder` au lieu de le traiter comme un MP4.
- Vérification : le venv du backend démarre FastAPI ; `/`, `/docs` et `/health`
  répondent HTTP 200. L'API a traité une capture WebM de deux secondes à 30
  images/s et renvoyé `Maty` (confiance 78,0 %). Le modèle audio ONNX a été
  chargé dans le navigateur et a généré 23 296 échantillons à 16 kHz.
- Le précédent essai sur
  `data/raw/signeur4/Miarahaba/01.mp4` renvoie toujours `Solosaina` à 54,6 % :
  l'infrastructure fonctionne, mais cette erreur de classification du modèle
  reste à résoudre séparément.
- À la demande de l'utilisateur, le compte à rebours avant l'enregistrement est
  réglé à trois secondes pour laisser le temps de préparer le signe.
