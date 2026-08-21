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
| E07 | EDA — nettoyage | `Neighborhood` : uniformisation de la casse et retrait du suffixe `NEIGHBORHOODS`. **19 modalités brutes ramenées à 13 quartiers réels** (`BALLARD`/`Ballard`, `NORTH`/`North`, `DELRIDGE`/`Delridge`/`DELRIDGE NEIGHBORHOODS`…) | 29 bâtiments étaient isolés dans six catégories fantômes de 1 à 9 individus. P3 encodait les 19 telles quelles, et le `BinaryEncoder` leur donnait des motifs de bits sans rapport : `Ballard` était aussi éloigné de `BALLARD` que n'importe quel autre quartier. Signal repoolé, 6 colonnes de moins |
| E08 | FE — fuite de données | **Les ratios `RatioElec` / `RatioGaz` / `RatioSteam` sont remplacés par trois booléens de présence** `HasElectricity` / `HasGas` / `HasSteam`. Les ratios se calculent depuis la consommation réelle : un utilisateur qui les connaît connaît déjà la réponse | Écart structurant du projet. La performance en labo va baisser, mais le modèle devient utilisable sur un bâtiment non relevé — ce qui est le cas d'usage. `HasElectricity` est conservée malgré 99,7 % de constance, par cohérence avec l'interface |
| E09 | FE — variable | `Age = 2016 − YearBuilt` supprimé, `YearBuilt` conservé brut | Aucun sur les métriques (transformation affine : arbres, régression linéaire et SVR y sont insensibles). Supprime un train/serve skew : `Age` dépend d'une année de référence que l'app aurait fait dériver chaque année |
| E10 | FE — composition d'usages | Les parts de surface par usage sont normalisées **sur la somme des GFA déclarées**, et non sur `PropertyGFATotal` comme dans P3. L'écrêtage à 1 disparaît | La couverture réelle allait de 0 à **6,43** (50 bâtiments au-dessus de 1,5) : la composition devient robuste à ces incohérences de déclaration. Surtout, elle correspond à ce que l'interface collecte — des pourcentages qui font 100 % |
| E11 | FE — encodage | `Neighborhood` en one-hot (13 colonnes) au lieu du `BinaryEncoder` de P3 (5 colonnes) | Supprime les relations numériques arbitraires entre quartiers et l'objet encodeur à persister pour servir. 13 modalités sur 1655 lignes ne justifient aucune compression |
| E12 | FE — variables écartées | `PrimaryPropertyType` non retenu (P3 l'avait encodé sans l'utiliser) ; `PropertyGFATotal` remplacé par son seul log ; `PropertyGFABuilding(s)` écarté | `PrimaryPropertyType` est à 62 % un renommage de `LargestPropertyUseType`, et sa seule information propre (Large Office vs Small- and Mid-Sized Office) est déjà portée par `logGFAtotal` — 21 colonnes one-hot dont plusieurs à moins de 15 exemples auraient ajouté du bruit. `PropertyGFABuilding(s)` vaut exactement `PropertyGFATotal − PropertyGFAParking` sur les 1656 lignes |
| E13 | FE — variable ajoutée | `ParkingRatio` = `PropertyGFAParking` / `PropertyGFATotal`, que P3 avait abandonné comme redondant avec `t_Parking` | Il ne l'est pas : corrélation de **0,49** seulement, et sur 332 bâtiments avec surface de parking contre 323 le déclarant comme usage, seuls 234 sont dans les deux cas. Candidat à confirmer en bivariée |
| E14 | FE — périmètre | Le bâtiment `496` est retiré du jeu d'entraînement : aucune répartition d'usage, donc composition 0/0 | 1655 lignes au lieu de 1656. C'est un point hors domaine — l'interface exige au moins un usage — qui aurait faussé l'évaluation s'il était tombé dans le jeu de test. Clôt le point laissé ouvert en EDA |
| E15 | FE — robustesse | Le mapping d'usages accueille `Small- and Mid-Sized Office`, et un usage ou un quartier inconnu lève une exception au lieu de produire une colonne fantôme `"ERROR"` | La correction E01 a introduit cette valeur : elle appartient au vocabulaire de `PrimaryPropertyType`, pas à celui des usages. P3 ne l'a jamais vue car il recopiait `Hotel`, présent dans les deux. En production, un bâtiment mal encodé est pire qu'une prédiction refusée |

---

## Écarts assumés, non corrigés

Relevés pendant le portage, laissés en l'état après décision :

| Étape | Constat | Décision |
|-------|---------|----------|
| EDA — univariée | `colnumlog` contient `SteamUse(kBtu)`, `NaturalGas(kBtu)` et `PropertyGFAParking`, qui comportent beaucoup de zéros. `log_scale=True` les écarte silencieusement du graphique : la distribution affichée n'est pas celle des données | Conservé — cosmétique, aucun effet sur les données exportées |
| FE — composition | Pour les 150 bâtiments (9,1 %) déclarant plus de trois usages, la composition d'entraînement est calculée sur une répartition tronquée aux trois principaux puis renormalisée, alors que l'interface en collectera une complète | Conservé — sans effet pour les 91 % restants, et la source ne publie pas mieux. À mentionner comme limite du modèle |
| Général | Commentaires de code en français, contrairement à la convention du projet | Conservé pour garder le portage comparable au notebook P3 |
