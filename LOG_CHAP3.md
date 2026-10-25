# Journal de révision du chapitre 3

## Sections révisées

- **3.1 Introduction** : annonce la campagne, ses axes d'évaluation et la
  distinction entre les résultats Colab et l'artefact TFLite déployé.
- **3.2 Protocole expérimental** : décrit huit configurations sans augmentation
  et deux entraînements augmentés, soit dix entraînements. Précise le split
  fixe, le rééchantillonnage temporel, les différences de dropout et les graines
  attestées.
- **3.3 Résultats détaillés** : rassemble les métriques communiquées et analyse
  la représentation, la longueur temporelle, la taille des modèles et
  l'augmentation sans inférence statistique ou causale non justifiée.
- **3.4 Confrontation** : compare les trois meilleurs scores observés et motive
  le choix de l'architecture de référence comme décision de conception, sans la
  présenter comme le meilleur score.
- **3.5 Analyse des erreurs** : remplace la comparaison de matrices obsolète par
  une matrice recalculée avec l'interpréteur TFLite actuellement déployé sur les
  100 séquences prétraitées du jeu de test.
- **3.6 Synthèse** : distingue les résultats de la campagne, le choix de
  conception et les performances de l'artefact TFLite présent dans le dépôt.
- **Chapitre 5, sections 5.3 et 5.4** : actualise les limites expérimentales et
  formule des perspectives directement liées aux observations.
- **Résumé, abstract et conclusion** : harmonisés avec les résultats de la
  campagne, les mesures du modèle déployé et les essais vidéo de l'API.
- **Chapitre 2** : distingue les hyperparamètres de la comparaison de référence
  des configurations de la campagne, précise les graines effectivement
  documentées et ne spécifie pas un GPU non attesté par les journaux.
- **Chapitre 4** : distingue l'artefact TFLite de la campagne Colab et précise
  que la capture d'interface n'est pas l'un des dix essais vidéo.
- **Listes des symboles et abréviations** : mises à jour pour les dimensions
  variables des entrées, LiteRT et CTC.

## Résultats et décisions consignés

- Les huit configurations sans augmentation et leurs métriques proviennent du
  rapport d'expériences Colab communiqué.
- Deux entraînements supplémentaires ont été confirmés par l'utilisateur :
  GRU et LSTM augmentés. L'exactitude du GRU passe de 92 % à 77 % ; celle du
  LSTM reste à 82 %. Le F1 macro du LSTM augmenté n'a pas été communiqué.
- Les comparaisons entre tailles de modèles sont nuancées, car le dropout varie
  également. Le score du GRU à 256 unités n'est pas présenté comme une preuve
  d'un surapprentissage.
- Les variantes temporelles de 15 et 45 frames sont créées à partir des
  séquences déjà ramenées à 30 frames. La variante à 45 frames ne restitue donc
  pas les images écartées lors du premier rééchantillonnage.
- La variante mains seules exclut les coordonnées du corps de l'entrée du
  classifieur, mais conserve la normalisation préalable par les épaules.
- Le modèle TFLite du dépôt a été exécuté sur `X_test` avec LiteRT : 90
  prédictions correctes sur 100. La nouvelle référence GRU Colab rapporte 92
  correctes sur 100. La provenance des poids ne permet pas d'établir que ces
  deux évaluations portent sur le même checkpoint.
- La nouvelle matrice TFLite a été produite à partir des prédictions de
  l'interpréteur sur `X_test`. Elle décrit le modèle déployé, pas les nouveaux
  modèles Colab.

## Points restant à clarifier

- Fournir ou archiver le checkpoint GRU Colab à 92 % et vérifier s'il correspond
  au modèle Keras et au TFLite utilisés par l'application.
- Obtenir les prédictions ou matrices de confusion des nouveaux modèles Colab
  à 92 % et 82 %, ainsi que leurs courbes d'apprentissage, pour remplacer les
  anciennes sorties si une analyse par classe de ces modèles est nécessaire.
- Obtenir le F1-score macro du LSTM augmenté, s'il a été calculé.
- Le F1-score macro du LSTM augmenté n'a pas été communiqué ; le mémoire
  l'indique comme non disponible au lieu de l'inférer à partir de l'exactitude.
- Les matrices de confusion et courbes des nouveaux modèles Colab ne sont pas
  disponibles. Le mémoire n'attribue donc pas aux nouveaux modèles les anciennes
  matrices ; la matrice incluse porte explicitement sur le TFLite déployé.
- La provenance des poids du checkpoint GRU Colab à 92 % et du TFLite déployé
  à 90 % reste à confirmer.

## Vérifications

- Compilation finale forcée avec `latexmk -g -pdf -interaction=nonstopmode
  main_francais.tex` depuis `memoire/` : réussie, 56 pages.
- La citation du glossaire de l'annexe II est routée vers la webographie :
  `GlossaireLSMMadagascar2019` figure dans `refs.aux` et non dans
  `main_francais.aux`. La passe BibTeX suivante ne signale plus de clé
  manquante dans la bibliographie principale.
- `git diff --check` : réussi.
- Matrice du TFLite : 100 séquences évaluées, 90 prédictions correctes.
