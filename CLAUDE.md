# OC_P13 — Règles du projet

Mise en production du modèle de prédiction énergétique de P3 (Seattle Building
Energy Benchmarking 2016). Plan de travail : `brief/plan_projet_technique.md`.

## Traçabilité des écarts avec P3

Ce projet **rejoue et fait diverger** le code de P3. Tout changement qui s'écarte
de P3 et touche aux données, aux features ou au modèle doit être consigné dans
`docs/ecarts_vs_P3.md` — une ligne par changement :

| ID | Étape | Changement (résumé court) | Impact attendu |

La ligne est ajoutée **au moment du changement**, pas après coup. Le fichier est
un journal : on y ajoute, on n'y réécrit pas l'historique.

**Pourquoi :** la section 2 du rapport de conduite de projet (audit de la
solution existante) doit justifier chaque écart. Git dit *quoi* a changé, ce
fichier dit *pourquoi* et *avec quelle conséquence attendue*.

Les constats relevés mais volontairement **non** corrigés vont dans la seconde
table du même fichier — un écart assumé et documenté vaut mieux qu'un écart
oublié.
