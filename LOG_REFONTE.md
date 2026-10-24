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
- L'étape 1 est validée. La taille du fichier modèle source (1,7 Mo) a été
  réduite de 85,5 % pour l'artefact TFLite.

## Étape 2 — Backend FastAPI avec TFLite

Non commencée. Vérifier d'abord que l'artefact TFLite est inclus dans Git malgré
les règles d'exclusion des modèles, puis recopier fidèlement le prétraitement
de `src/extract.py` et valider l'API sur une vidéo du corpus.
