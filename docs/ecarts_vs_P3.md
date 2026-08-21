# Écarts avec le projet P3

Ce projet reprend les notebooks de P3 (*Anticipez les besoins en consommations
de bâtiments*, Seattle 2016) et les porte en production. Chaque divergence par
rapport à P3 est consignée ici : une ligne, l'étape concernée, le changement en
résumé, l'impact attendu.

**Pourquoi :** au moment de rédiger le rapport de conduite de projet, la section 2
(audit de la solution existante) doit justifier chaque écart. Ce fichier est la
source de cette section — git dit *quoi* a changé, ce tableau dit *pourquoi* et
*avec quelle conséquence*.

**Règle :** la ligne est ajoutée au moment du changement, pas après coup.

---

| ID | Étape | Changement | Impact attendu |
|----|-------|-----------|----------------|
| E01 | EDA — nettoyage | `LargestPropertyUseType` / `LargestPropertyUseTypeGFA` : les bâtiments `25568` et `25711` reçoivent leurs propres valeurs. P3 leur recopiait celles du bâtiment `21103` (Hotel, 61 721 sq ft) — erreur de copier-coller | 2 lignes sur 1656. Corrige les ratios de surface qui seront calculés en FE pour ces deux bâtiments. Effet global sur les métriques négligeable, mais deux observations fausses en moins |
| E02 | EDA — imputation | `NumberofFloors` : les 16 valeurs à 0 sont remplacées par la **médiane** des valeurs non nulles, calculée (2), au lieu de la constante `4` codée en dur dans P3. La distribution est très asymétrique (moyenne 4,17 / médiane 2) : sur ce type de variable, la médiane est l'imputation adaptée | 16 lignes passent de 4 à 2 étages — c'est le plus gros écart de valeurs du portage. Impact attendu faible sur les métriques (1 % des lignes), mais l'imputation n'est plus tirée vers le haut par les tours du centre-ville. La colonne reste en `int64`, la valeur se recalcule si les données changent |
| E03 | EDA — nettoyage | Suppression de la ligne imputant les `NaN` de `NumberofBuildings` : la colonne n'en contient aucun, l'instruction était morte | Aucun. Lisibilité seulement |
| E04 | EDA — périmètre | Le notebook s'arrête au nettoyage et exporte `data/dfclean.csv` (21 colonnes non transformées, sans colonne d'index). P3 exportait `dftclean2.csv` déjà transformé par la FE | Aucun sur les valeurs. Permet de rejouer la même FE à l'entraînement et à l'inférence via `src/features.py` |
| E05 | EDA — imputation | `NumberofBuildings` : règle P3 conservée (0 → 1, 52 lignes), mais la justification passe de « proche de la moyenne » à une vérification factuelle — aucun de ces bâtiments n'est de type `Campus`, et leur GFA médiane (55 561 sq ft) est proche des mono-bâtiments (48 020) et très loin des multi-bâtiments (111 445) | Aucun changement de valeur. Hypothèse défendable en soutenance au lieu d'une constante arbitraire |
| E06 | EDA — environnement | pandas 3.0.5 au lieu de pandas 2.x : l'upcast implicite `int64` → `float64` sur `.loc` est devenu une erreur dure là où pandas 2.x se contentait d'un `FutureWarning`. Rencontré en tentant l'imputation par la moyenne (4,17) ; sans objet depuis le passage à la médiane (2, entière) | Aucun sur les valeurs. À garder en tête pour la FE : toute imputation ou transformation non entière sur une colonne entière devra convertir explicitement |

---

## Écarts assumés, non corrigés

Relevés pendant le portage, laissés en l'état après décision :

| Étape | Constat | Décision |
|-------|---------|----------|
| EDA — univariée | `colnumlog` contient `SteamUse(kBtu)`, `NaturalGas(kBtu)` et `PropertyGFAParking`, qui comportent beaucoup de zéros. `log_scale=True` les écarte silencieusement du graphique : la distribution affichée n'est pas celle des données | Conservé — cosmétique, aucun effet sur les données exportées |
| EDA — nettoyage | Le bâtiment `496` garde ses `NaN` sur `LargestPropertyUseType` / GFA : la répartition des usages est introuvable | Conservé (choix P3) — 1 ligne sur 1656, les modèles doivent gérer la valeur manquante |
| Général | Commentaires de code en français, contrairement à la convention du projet | Conservé pour garder le portage comparable au notebook P3 |
