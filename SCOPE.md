# SCOPE.md — Cadrage du périmètre du projet

## 1. Titre de travail
Traduction de signes isolés de la Langue des Signes Française (LSF) vers
texte et audio : comparaison contrôlée d’architectures récurrentes
(LSTM vs GRU) sur caractéristiques de pose extraites par MediaPipe
Holistic, sous contrainte de ressources computationnelles limitées.

## 2. Question de recherche
Dans quelle mesure une architecture récurrente (LSTM vs GRU) entraînée sur
des caractéristiques de pose (MediaPipe Holistic, vecteur 1662D, séquences
rééchantillonnées à 30 frames) permet-elle une traduction fiable de signes
isolés de la LSF vers du texte, dans un contexte de ressources
computationnelles limitées et avec un vocabulaire fermé ?

## 3. Périmètre — INCLUS
- Langue : LSF.
- Type de traduction : signe isolé → un mot (pas de séquences multi-signes).
- Vocabulaire : 12 signes, liste fermée (voir §6).
- Nombre de signeurs : 1 (limite assumée explicitement dans le rapport).
- Représentation : MediaPipe Holistic, vecteur 1662D par frame,
  séquence rééchantillonnée à T = 30 frames.
- Modèles comparés : LSTM (baseline) et GRU (comparaison contrôlée à
  hyperparamètres identiques).
- Extension optionnelle (niveau 2) : BiLSTM ou baseline agrégée (moyenne
  temporelle + MLP), activée seulement si le niveau 1 est stable.
- Sortie : mot prédit (texte) + synthèse vocale (gTTS).
- Application web de démonstration (Gradio) : upload vidéo → texte + audio,
  guide d'utilisation documenté.
- Entraînement : Google Colab (GPU T4), machine locale réservée au
  développement et à l'extraction MediaPipe.

## 4. Périmètre — EXCLU (assumé comme limite)
- Séquences multi-signes et traduction de phrases.
- Multi-signeurs (généralisation inter-signeurs).
- Normalisation linguistique officielle de la LSF (le vocabulaire est
  celui que nous définissons, pas une norme établie).
- Modèles Transformer / seq2seq avec attention.
- GCN et architectures hybrides lourdes.
- Reconnaissance continue en temps réel sur flux long.
- Déploiement en production (cloud, multi-utilisateurs, authentification).

## 5. Contraintes techniques
- Machine locale : Lenovo ThinkPad L440, i5 4e génération, 4 Go RAM,
  pas de GPU dédié → pas d'entraînement local.
- Entraînement : Google Colab, GPU T4, dataset de keypoints sur Drive.
- Versionnage : GitHub, dépôt de travail `Rinoh-coder/Projet-1`.
- Template LaTeX : copié depuis `memoire/` du dépôt de référence
  `traduction-de-langue-des-signes-Projet-1-landmarks-LSFB`, non modifié
  dans sa structure.
- Rédaction : assistée par Manus (connecteur GitHub), validée par
  l'étudiant avant merge.

## 6. Vocabulaire retenu (12 signes)
[À COMPLÉTER PAR L'ÉTUDIANT — liste fermée de 12 signes LSF, justifiée
par leur distinctivité visuelle]
Liste proposée (à valider ou remplacer) :
1. bonjour
2. merci
3. oui
4. non
5. un
6. deux
7. trois
8. quatre
9. cinq
10. eau
11. manger
12. aider

## 7. Protocole de captation
[À COMPLÉTER PAR L'ÉTUDIANT — conditions de tournage]
Paramètres fixés :
- 12 signes × 10 répétitions = 120 séquences minimum.
- Durée par séquence : 2 secondes.
- Compte à rebours de 3 secondes avant chaque enregistrement.
- Caméra fixe, éclairage constant, arrière-plan neutre.
- Résolution : 640×480 ou 1280×720, 25–30 fps.
- Sauvegarde : data/raw/{signe}/{rep:02d}.mp4.
- Script : src/capture.py.

## 8. Pipeline technique (résumé)
1. Captation automatisée → data/raw/
2. Extraction MediaPipe Holistic → (T, 1662) par séquence
3. Rééchantillonnage à T = 30 frames (interpolation linéaire)
4. Normalisation référencée aux épaules
5. Splits 70/15/15 par séquence (jamais par frame) → splits.json
6. LSTM baseline → models/lstm_best.h5
7. GRU (mêmes hyperparamètres) → models/gru_best.h5
8. Analyse comparative (accuracy, F1 macro, matrices de confusion)
9. Application Gradio : upload vidéo → texte + audio gTTS
10. Test de robustesse sur séquences nouvelles
11. Rapport LaTeX + slides

## 9. Critères de succès
- Pipeline complet exécutable de bout en bout sans intervention manuelle.
- Comparaison LSTM vs GRU documentée avec métriques et matrices.
- Application testée sur au moins 5 séquences non vues, avec tableau de
  résultats end-to-end et taux de succès documenté.
- Guide d'utilisation de l'application livré.
- Test de robustesse exécuté, échecs rapportés sans masquage.
- Rapport documentant explicitement toutes les limites du §4.
- Reproductibilité : un tiers peut refaire le pipeline via le README.

## 10. Ce que ce projet n'est PAS
- Pas une percée de l'état de l'art.
- Pas un système de traduction de LSF généraliste.
- Pas un produit déployable.
C'est un proof-of-concept reproductible, dont la valeur scientifique
réside dans la rigueur méthodologique et la transparence des limites.

## 11. Références candidates (à valider par fiches de lecture)
- Braffort (2008) — HDR, traitement automatique de la LSF, LIMSI.
- Ouakrim et al. (2023) — modèle multistream, pose 3D, BiRNN + CTC, LSF.
- Takouchouang (2025) — CNN-LSTM + MediaPipe, confusions sur gestes proches.
- Camgoz et al. — PHOENIX-2014T (benchmark SLT).
- Documentation officielle MediaPipe Holistic.
- Documentation TensorFlow/Keras (LSTM, GRU).
- Documentation gTTS.

## 12. Décisions figées
Toute modification de ce document doit être justifiée dans LOG.md,
commitée avec un message explicite, et référencée dans la PR correspondante.

Date de figement : [à dater au premier commit de SCOPE.md]
