# Journal de révision du mémoire

Les révisions ci-dessous ont été appliquées séparément après validation.

## Points traités

### Point 6 — Langue / Language

Aucune modification nécessaire. Le glossaire emploie déjà « Langue des Signes
Française » et « Langue des Signes Malgache ». Les occurrences anglaises dans
l'abstract, les sigles et les titres bibliographiques ont été conservées.

### Point 1 — Cohérence sur les signeurs

Le résumé français, l'abstract anglais, l'introduction et la conclusion précisent
que trois des quatre signeurs sont affectés à l'entraînement et à la validation,
tandis qu'un signeur est réservé au test. Cette formulation rend explicite la
séparation déjà décrite dans la méthodologie.

### Point 7 — Source des illustrations de l'annexe II

Une attribution commune a été ajoutée aux dix illustrations de l'annexe II et une
référence a été ajoutée à la webographie. Elle identifie le *Petit glossaire de la
langue des signes à Madagascar*, travail collectif d'étudiants d'Aix-Marseille
Université sous la direction de Mélanie Hamm, et reconnaît la contribution
d'Andoniaina Raphaël Rabarijohn Lucien. La dernière mise à jour indiquée dans le
document est le 14 septembre 2019. L'uploadeur Scribd est mentionné comme diffuseur,
pas comme auteur.

### Point 2 — Conditions de capture

Le chapitre 2 précise que les quatre signeurs comprennent trois hommes et une
femme, que les captures ont eu lieu dans deux pièces sous un éclairage naturel
constant et que le signeur réservé au test est un homme filmé dans la seconde
pièce. Le chapitre 5 a été ajusté pour distinguer ces conditions observées des
paramètres que le script ne contrôle pas.

### Point 5 — Vocabulaire d'ingénierie logicielle

Le chapitre 4 explicite la séparation des responsabilités, la distinction entre
l'artefact Keras d'entraînement et l'artefact TFLite de déploiement, ainsi que les
mécanismes de programmation défensive. Les dix essais sur vidéos sont qualifiés de
tests de bout en bout du parcours d'inférence côté serveur ; leur périmètre ne
comprend pas toute la chaîne navigateur et synthèse audio.

### Point 4 — Analyse des erreurs

Le chapitre 3 formule au conditionnel des hypothèses pour les confusions observées
sur Tsia, Maty et Miarahaba. Les associations citées correspondent aux matrices
LSTM et GRU. L'ajout de points du visage est présenté comme une piste à évaluer,
et non comme un résultat établi.

### Point 3 — Analyse des études comparées

Après le tableau comparatif du chapitre 1, quatre courts paragraphes présentent
les études d'Ilham et al., Gonfa et al., Alyami et al. et Maquera et al. Chaque
paragraphe indique l'approche étudiée, son apport à la comparaison et une
différence avec le présent travail. Aucun score n'est attribué à Maquera et al.,
car le tableau n'en rapporte pas.

## Vérifications et éléments à poursuivre

- Compilation LaTeX réussie lors de cette révision ; le document comptait alors
  49 pages. La compilation finale après harmonisation, consignée ci-dessous,
  compte 56 pages.
- `git diff --check` réussi.
- Les occurrences de « Language » restantes sont en anglais et appropriées ; le
  glossaire français emploie « Langue ».
- La séparation des quatre signeurs est cohérente dans le résumé, l'introduction,
  la méthodologie et la conclusion.
- Le tableau de non-détection a été calculé sur les 24 000 frames des 400 vidéos,
  avec MediaPipe Holistic 0.10.14 en mode dynamique. Il indique l'absence de
  détection des groupes de pose et de mains, et non des points individuels.

## Analyses complémentaires du corpus

- Vérification structurelle des 400 vidéos : chaque fichier est lisible et
  contient 60 images en 640 × 480 à 30 images/s, soit deux secondes. Les quatre
  signeurs ont dix répétitions par classe.
- Audit de `meta.csv` et `splits.json` : chaque classe comporte 21 séquences
  d'entraînement, 9 de validation et 10 de test ; le test contient uniquement
  `signeur4`. Les fichiers de caractéristiques ont les formes attendues et ne
  contiennent pas de valeurs non finies.
- Contrôle numérique de la normalisation : sur 12 000 images, le centre des
  épaules est nul et leur distance normalisée égale un, aux erreurs d'arrondi
  float32 près.
- Analyse du jitter : sur les 210 séquences d'entraînement, le rapport médian
  entre la distance des extrémités et la médiane des distances entre images
  consécutives est de 4,10 ; cette distance d'extrémité est supérieure dans
  206 séquences. Le résultat signale un risque de discontinuité du décalage
  circulaire, sans établir qu'il cause à lui seul la baisse des performances.
- Le script `src/audit_corpus.py` reproduit les contrôles de métadonnées, de
  partitions, de propriétés vidéo, de tenseurs, de normalisation et de
  discontinuité temporelle sans modifier les données. Le tableau de détection
  MediaPipe provient du balayage détaillé des 24 000 images.

## Vérification finale après harmonisation

- Compilation forcée puis stabilisée avec `latexmk -g -pdf
  -interaction=nonstopmode main_francais.tex` depuis `memoire/` : réussite,
  PDF de 56 pages.
- La citation du glossaire des signes de l'annexe II est routée vers la
  webographie. Après la seconde passe BibTeX, la clé
  `GlossaireLSMMadagascar2019` apparaît dans `refs.aux` et non dans
  `main_francais.aux` ; aucun avertissement BibTeX correspondant ne subsiste.
- `git diff --check` : réussi.
- La compilation conserve des avertissements typographiques `Underfull \hbox`,
  des avertissements `caption` pour les légendes de l'annexe II et un
  avertissement Babel concernant l'option historique `francais`. Ils ne
  bloquent pas la génération du PDF et ne sont pas liés à l'harmonisation des
  résultats.
- Le logo ESPA de la couverture et de la page du jury utilise désormais le
  nouveau fichier `figures/images du memoire/logoESPA.jpg`. Les deux
  encadreurs portent la distinction « Dr. » dans ces pages. Une recompilation
  confirme la prise en compte du nouveau fichier et conserve un PDF de
  56 pages.

## Harmonisation finale du modèle déployé

Sur instruction explicite de l'utilisateur : le modèle exécuté par l'application
est le modèle de référence GRU pose+mains (30 frames, 128 unités, 137 610
paramètres, exactitude de 92,0 %) converti au format TFLite. La distinction
narrative antérieure entre le checkpoint expérimental Colab (92 %) et un
artefact TFLite distinct (90 %) est donc retirée du rapport.

### Modifications

- `couverture/2-encadreurs.tex` : distinction « M. » pour
  RANDRIAMASINORO Njakarison Menja (vérifiée ; « Dr. » conservé pour
  RAJONIRINA Solofanja Jeannie).
- `parties/03-resume.tex` : résumé et abstract réécrits en formulations
  simples et claires ; suppression de la phrase sur les artefacts distincts
  (et de son équivalent anglais) ainsi que du « 90 résultats corrects sur
  100 » ; le résumé présente désormais le GRU retenu à 92,0 %, sa conversion
  TFLite et l'application web, avec l'essentiel des limites.
- `parties/chapitre3.tex` : la figure de la matrice de confusion de l'ancien
  artefact (90 %) et son analyse par classe sont retirées ; la section devient
  « Analyse des erreurs en conditions d'utilisation » et s'appuie sur les dix
  essais vidéo de l'endpoint (chapitre 4) avec des hypothèses au conditionnel.
  La confrontation justifie désormais le choix du GRU de référence par la
  comparaison contrôlée avec le LSTM (tous hyperparamètres identiques), la
  cohérence du pipeline pose+mains et la continuité entraînement/inférence.
- `parties/chapitre4.tex` : le paragraphe de conversion indique que l'artefact
  TFLite exécuté par le backend est la conversion du modèle de référence
  évalué à 92,0 % ; retrait des mentions « n'est pas déployé » et de la
  correspondance de checkpoints.
- `parties/chapitre5.tex` : retrait des évaluations séparées à 90 % ; la
  perspective de reproductibilité porte sur l'archivage de la chaîne de
  production de l'artefact (checkpoint, script et journal de conversion) et
  sur la vérification comparée Keras/TFLite sur les données de test. La
  perspective finale renforce la traduction continue : corpus plus ample en
  quantité et en qualité, annotations fiables et ressources humaines et
  linguistiques dédiées.
- `parties/07-conclusion.tex` : le GRU de référence est présenté comme le
  modèle retenu et déployé après conversion TFLite ; les essais vidéo de
  l'endpoint (7/10) restent consignés.
- `parties/06-introduction.tex` : le plan des chapitres mentionne l'analyse
  des erreurs en conditions d'utilisation au lieu d'un « test de robustesse »
  obsolète.
- Aucun F1-score du LSTM augmenté n'a été ajouté au-delà de la valeur
  0,7840 déjà présente dans le tableau du chapitre 3 ; la note d'indisponibilité
  avaient déjà été supprimée dans une révision antérieure.

### Vérifications

- Compilation avec `latexmk -g -pdf -interaction=nonstopmode
  main_francais.tex` : réussie, PDF de 57 pages.
- Aucune erreur LaTeX, aucune référence ou citation non définie dans le log.
- `git diff --check` : réussi.
- Recherche dans `memoire/parties/` : plus aucune occurrence de
  « 90/100 », « n'est pas déployé » ou de correspondance de checkpoints
  à établir, hors la perspective d'archivage volontairement conservée.

### Dédoublonnage chapitres 3 et 4

- `parties/chapitre3.tex` : la section « Analyse des erreurs en conditions
  d'utilisation » conserve le résultat (7/10) et les trois confusions,
  nécessaires à son interprétation, mais s'ancre désormais directement dans
  le tableau des essais de l'endpoint du chapitre 4
  (`Tableau~\ref{tab:tests-integration}`) au lieu de reformuler le protocole.
- `parties/chapitre4.tex` : la section « Tests d'intégration et validation »
  reste factuelle (méthode, tableau, portée du parcours serveur validé) et
  renvoie au chapitre 3 pour l'analyse de l'écart avec l'évaluation hors
  ligne ; la phrase dupliquée sur la portée des essais a été retirée de ce
  chapitre.
- Compilation revalidée après dédoublonnage : PDF de 57 pages, aucune
  erreur, aucune référence non définie.

### Audit complet de cohérence

- `parties/chapitre5.tex` : la phrase faisant de Tsia et Maty les signes les
  plus difficiles se référait à l'ancienne matrice de confusion (90 %)
  retirée ; elle est remplacée par une formulation fondée sur les erreurs des
  essais vidéo analysées au chapitre 3 (Marenina, Maty, Miarahaba).
- `parties/chapitre3.tex` : les effectifs de classe par partition sont
  corrigés en 21 (entraînement), 9 (validation), 10 (test) par signe, en
  cohérence avec le chapitre 2 et l'annexe « Corpus LSM » ; double
  « ensuite » de l'annonce du plan simplifié.
- Contrôles arithmétiques effectués : 92/100 séquences ; écarts de 2 et 1
  séquences entre les trois meilleures configurations ; 251 344 octets =
  245,5 Kio et 1 677 751 octets = 1,60 Mio, réduction de 85,0 % ; 99 594
  paramètres = −27,6 % par rapport à 137 610 ; 225 = 33 x 3 + 21 x 3 x 2 ;
  225 - 99 = 126 coordonnées pour les mains seules ; 1 629 = 225 + 468 x 3,
  ratio 7,24 ; 12 000 = 400 x 30 frames et 24 000 = 400 x 60 ; répartition
  annexe 3 x 7 x 10 = 210, 3 x 3 x 10 = 90, 10 x 10 = 100.
- Vérification exhaustive : chiffres identiques dans le résumé, les
  chapitres 3 à 5 et la conclusion ; liste des erreurs identique dans les
  trois sections concernées ; aucune mention de Tsia issue de l'ancienne
  matrice ; compilation finale propre (57 pages, zéro erreur, zéro
  référence non définie).

## Mise à jour : modèle final déployé (réentraîné avec unroll=True)

Sur instruction explicite de l'utilisateur : le modèle GRU final,
réentraîné avec `unroll=True` pour rendre la conversion TFLite possible,
atteint 94,0 % d'exactitude sur le jeu de test et est le modèle déployé
dans l'application ; son artefact TFLite mesure 247 Ko. La remarque
rédactionnelle sur le 94,0 % (visant la phrase du §3.4) a été traitée en
ajoutant la sous-section "Effet de la contrainte de déploiement" dans
l'insertion de la 3.3.5, là où elle est justifiée par le flux du texte.

### Modifications (périmètre minimal)

- `parties/03-resume.tex` : le résumé français distingue désormais le
  LSTM (82,0 %), le GRU expérimental (92,0 %) et le modèle final GRU
  réentraîné sous contrainte de conversion (94,0 %), avec l'artefact à
  247 Ko ; l'abstract (94,0 %, 247 KB) reste aligné.
- `parties/chapitre3.tex` : le protocole annonce le réentraînement sous
  contrainte ; nouvelle sous-section 3.3.5 "Effet de la contrainte de
  déploiement" (label `sec:deploiement-tflite`) : réentraînement avec
  `unroll=True`, conversion sans opérateur Flex, modèle final à
  94,0 % ; la section 3.4 (choix du modèle retenu) se termine par le
  renvoi vers cette sous-section et le score de 94,0 % ; l'analyse des
  erreurs (3.5) rapporte 92,0 % pour le modèle de référence et
  94,0 % pour le modèle final déployé, avec le renvoi ; la synthèse
  (3.6) mentionne le modèle final à 94,0 % déployé et reformule
  l'argument "mains seules" en cohérence (même score de 94,0 % dans la
  campagne, avantage non établi comme significatif).
- `parties/chapitre4.tex` : le paragraphe de conversion indique le
  réentraînement avec `unroll=True`, le modèle final à 94,0 % et l'artefact
  TFLite à 247 Ko (contre le modèle Keras de 1 677 751 octets, environ
  1,60 Mio) ; le tableau des choix technologiques cite l'artefact à 247 Ko.
- `parties/chapitre5.tex` : la limite sur le modèle indique 94,0 % pour le
  modèle final déployé ; la limite méthodologique conclut que
  l'exactitude rapportée concerne bien le modèle déployé (modèle final
  réentraîné avec `unroll=True`, 94,0 %).
- `parties/07-conclusion.tex` : la conclusion relate le réentraînement
  avec `unroll=True`, le modèle final à 94,0 % et son artefact à 247 Ko.

### Vérifications

- Compilation avec `latexmk -pdf -interaction=nonstopmode
  main_francais.tex` depuis `memoire/` : réussie, PDF de 57 pages,
  aucune référence non définie, aucun label instable.
- `git diff --check` : réussi.
- Recherche de résidus : plus aucune occurrence de "245,5" ni de
  "251 344" ; les mentions de 92,0 % restent uniquement pour le modèle
  de référence initial et le GRU expérimental, jamais pour le modèle
  déployé.
- `figures/images du memoire/matrice_confusion_gru_final.png`
  (43 781 octets, 913 x 790) : insérée comme nouvelle Figure 3.1 du
  chapitre 3 (section "Analyse des erreurs en conditions d'utilisation"),
  avec la légende demandée ; le texte de la section 3.5 est remplacé par
  la formulation fournie (94,0 %, erreurs principalement Tsia, Maty et
  Miarahaba, cohérence avec la littérature) ; l'analyse des essais vidéo
  conserve le renvoi au Tableau 4.x avec l'évaluation hors ligne mise à
  jour à 94,0 %.
- `backend/models/model_gru.tflite` : l'artefact fourni (252 712 octets,
  soit 247 Ko) remplace l'ancien (251 344 octets) et correspond au
  chiffre cité dans le rapport ; committé séparément.
- Commits atomiques d93a364 (figure), 78bd03a (artefact TFLite) et
  3bfd296 (textes) poussés sur la branche.
