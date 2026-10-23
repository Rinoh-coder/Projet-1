# Application web de traduction de la langue des signes en audio

Ce dossier contient la version LaTeX du rapport du projet. Le template ESPA a été importé depuis `njakarison/Latex_Template` au commit source `9d8a606`.

## Périmètre

Le document porte exclusivement sur la reconnaissance de glosses LSFB isolées à vocabulaire contrôlé. Il décrit l'application web, le modèle Random Forest, l'interface HTML/CSS/JavaScript et le backend FastAPI. HTTP est utilisé pour le contrat d'inférence reproductible ; WebSocket est documenté comme extension pour un futur flux continu après validation causale.

## Compilation

Depuis ce dossier :

```bash
latexmk -pdf -interaction=nonstopmode -jobname=Rapport_memoire_projet_tutore main_francais.tex
```

À défaut de `latexmk`, exécuter plusieurs passes de `pdflatex` et `bibtex` afin de résoudre la table des matières et les références. Les fichiers de compilation doivent rester ignorés par Git.

## Organisation

- `main_francais.tex` : point d'entrée ;
- `couverture/` : page de garde, encadreurs et page de jury ;
- `parties/` : pages préliminaires, chapitres, conclusion et annexe ;
- `biblio/` : bibliographie scientifique et webographie ;
- `figures/` : logos et futures figures du mémoire ;
- `lib/` : styles fournis par le template ESPA.

Les fichiers Markdown dans `../docs/` restent la base de travail détaillée pour les expériences et la traçabilité. La migration LaTeX est progressive : aucun chiffre ne doit être ajouté au mémoire sans correspondance avec une expérience, un script, un commit ou une source identifiée.

## Informations à compléter

La date de soutenance, la date de dépôt et la composition officielle du jury sont encore des placeholders. Elles doivent être renseignées après confirmation institutionnelle.
