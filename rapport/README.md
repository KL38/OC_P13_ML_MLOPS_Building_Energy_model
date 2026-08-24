# Rapport de conduite de projet — OC_P13

## Mettre le projet sur Overleaf

1. Zipper **le contenu de ce dossier** (`main.tex` doit être à la racine du zip,
   pas dans un sous-dossier).
2. Sur Overleaf : **New Project → Upload Project**.
3. Vérifier que le document principal est `main.tex` et que le compilateur est
   **pdfLaTeX** (Menu → Compiler). C'est le réglage par défaut.

Aucun paquet à installer : tout ce qui est utilisé est dans la distribution
TeX Live standard d'Overleaf.

## Organisation

| Fichier | Contenu |
|---|---|
| `main.tex` | Page de titre, sommaire, appel des sections |
| `preambule.tex` | Paquets, couleurs, styles de titre, environnement `constat` |
| `sections/00_synthese.tex` | Synthèse d'une page |
| `sections/01_contexte.tex` | 1. Contexte et analyse des besoins |
| `sections/02_audit.tex` | 2. Audit de la solution existante |
| `sections/03_solution.tex` | 3. Solution technique cible |
| `sections/04_appui.tex` | 4. Appui stratégique et méthodologique |
| `sections/05_controle.tex` | 5. Contrôle et suivi |
| `sections/06_conclusion.tex` | 6. Conclusion et recommandations |
| `sections/07_annexes.tex` | 7. Annexes, dont le journal des 16 écarts |
| `figures/` | Captures d'écran à ajouter |

La structure des sections 1 à 7 suit exactement le template OpenClassrooms.

## Conventions

- `\begin{constat}{Titre}` : encadré à filet bleu, pour les constats qui doivent
  survivre à une lecture en diagonale.
- `\code{...}` : nom de fichier ou de variable.
- Les deux cibles du modèle gardent les couleurs de l'application : bleu pour
  l'énergie, orange pour les émissions.
- Les nombres passent par `\num{}` (siunitx) pour l'espace fine insécable.

## À faire avant remise

- [ ] Insérer les captures listées en fin d'annexe
- [ ] Supprimer la sous-section « Captures à insérer » de l'annexe
- [ ] Vérifier la date de la page de titre (`\today` se met à jour tout seul)
