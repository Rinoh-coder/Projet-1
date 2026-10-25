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

- Compilation LaTeX réussie avec `latexmk -pdf -interaction=nonstopmode
  main_francais.tex` ; le document généré compte 49 pages.
- `git diff --check` réussi.
- Les occurrences de « Language » restantes sont en anglais et appropriées ; le
  glossaire français emploie « Langue ».
- La séparation des quatre signeurs est cohérente dans le résumé, l'introduction,
  la méthodologie et la conclusion.
- Le tableau de non-détection a été calculé sur les 24 000 frames des 400 vidéos,
  avec MediaPipe Holistic 0.10.14 en mode dynamique. Il indique l'absence de
  détection des groupes de pose et de mains, et non des points individuels.
