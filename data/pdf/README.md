# data/pdf/ — document source du projet

Ce dossier contient **un seul** PDF technique, utilisé comme unique source du
RAG (Partie I). Son contenu réel n'est jamais commité (`.gitignore`) : seul ce
README reste versionné.

## Critères de sélection

- Domaine maintenance machine (manuel technique, procédure, notice
  d'entretien...).
- Contient du texte natif **et** des images **et** des tableaux (pas un PDF
  scanné sans OCR : `scripts/inspect_pdf.py` rejette les pages sans texte
  natif).
- Taille raisonnable : idéalement 20 à 80 pages.
- Licence compatible avec un usage académique (document public, notice
  constructeur librement diffusable, ou document dont vous détenez les
  droits).

## Où le déposer

Placez le fichier choisi dans ce dossier, par exemple :

```
data/pdf/document.pdf
```

puis vérifiez qu'il convient :

```bash
python scripts/inspect_pdf.py data/pdf/document.pdf
```

Le script affiche un résumé par page (texte, images, tableaux) et un
verdict `suitable` / `not suitable` (texte natif requis, >= 3 images et
>= 2 tableaux sur l'ensemble du document). Ajustez `PDF_PATH` dans `.env` pour
que les milestones suivants pointent vers ce fichier.
