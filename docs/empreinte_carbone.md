# Empreinte carbone du benchmark — ce que la mesure a tranché

**Destination : rapport, section « choix du modèle » et section « limites ».**

CodeCarbon a été branché non pour afficher un chiffre vert, mais pour faire
entrer le coût de calcul dans l'arbitrage entre modèles. Cette note dit ce que
la mesure a effectivement décidé, et ce qu'elle ne permet pas de décider.

## Protocole

CodeCarbon 3.3.0, mode `machine`, un `start_task` / `stop_task` par couple
(cible × modèle) — soit 12 tâches par run, mesurées séparément. Deux runs
complets : `seattle-booleens` (pipeline retenu) et `seattle-ratios_P3`
(reproduction de la fuite de P3, pour mesurer son coût en performance).

Machine : AMD Ryzen 5 7600X (12 threads), 32 Go, RTX 5070. Réseau français,
région Rhône-Alpes.

Contrôle de cohérence : la somme des 12 tâches vaut 3,7058 Wh, le compteur
global du run 3,7058 Wh. Le découpage par tâche ne perd rien.

## Ce que chaque modèle a coûté

Run `seattle-booleens`, cible **énergie** — R² hors fold en regard :

| Modèle | Durée | Énergie | CO₂eq | R²_log |
|---|---|---|---|---|
| Dummy | 0,00 s | 0,083 Wh* | 4,63 mg* | −0,000 |
| GradientBoosting | 0,48 s | **0,016 Wh** | 0,88 mg | **0,7447** |
| RandomForest | 1,38 s | 0,058 Wh | 3,26 mg | 0,7251 |
| HistGradientBoosting | 1,83 s | 0,057 Wh | 3,18 mg | 0,7322 |
| CatBoost | 3,71 s | 0,156 Wh | 8,73 mg | 0,7435 |
| TabPFN | 38,05 s | **1,586 Wh** | 88,9 mg | 0,7643 |

Cible **émissions** — mêmes ordres de grandeur : GradientBoosting 0,015 Wh
(R² 0,7195), CatBoost **0,147 Wh** (R² 0,7234, modèle retenu), TabPFN
**1,516 Wh** (R² 0,7445).

\* valeur non fiable, voir « ce que la mesure ne vaut pas ».

## Trois conclusions

### 1. TabPFN : 84 % du budget pour 2 points dans le bruit

À lui seul, TabPFN consomme **83,7 %** de l'énergie du benchmark complet
(3,10 Wh sur 3,71). Face au modèle retenu sur la même cible, il coûte **101 ×**
plus cher en énergie et **78 ×** plus long, pour un gain de **+1,96 point** de R²
sur l'énergie et **+2,11 points** sur les émissions — un écart du même ordre que
la variabilité d'échantillonnage sur 331 bâtiments de test.

TabPFN était déjà écarté pour sa licence (`tabpfn-3-license-v1.0` : évaluation
autorisée, production interdite). La mesure lui donne un **second motif de rejet,
indépendant du premier** : même sans obstacle juridique, le rapport coût/gain ne
le justifiait pas. C'est le seul arbitrage que CodeCarbon a réellement tranché
dans ce projet, et il vaut d'être dit tel quel.

Le couple retenu — GradientBoosting (énergie) + CatBoost (émissions) — représente
**4,4 %** de l'énergie du benchmark.

### 2. À cette échelle, l'entraînement ne pèse rien — l'hébergement, si

Le benchmark complet, deux runs et douze modèles, a consommé **7,3 Wh** et émis
**0,41 g de CO₂eq**. C'est une ampoule LED de 10 W allumée 44 minutes. Aucune
décision sérieuse ne peut se fonder sur un budget pareil : à cette échelle,
l'empreinte de l'entraînement n'arbitre rien **en absolu**. Elle n'arbitre que
par les **rapports** entre modèles (conclusion 1).

Le vrai poste, c'est l'hébergement. Un Space `cpu-basic` maintenu éveillé
consomme, à supposer 10 W de puissance moyenne, environ **88 kWh par an** — soit
**12 000 fois** le benchmark complet. Le modèle, lui, sera réentraîné une fois
par an, au rythme de publication du benchmark de Seattle.

**Décision qui en découle :** laisser actif le passage en veille automatique du
Space après inactivité, et ne pas le contourner par un keep-alive externe.
Le confort de démarrage instantané ne vaut pas de faire tourner un conteneur
en continu pour un usage épisodique. (Arbitrage inverse de celui pris sur
OC_P8, où la veille de la base cassait la fonctionnalité — ici elle ne coûte
qu'un temps de démarrage.)

### 3. Le mix électrique pèse plus lourd que le choix du modèle

Le facteur de conversion mesuré ici est de **56 g CO₂ / kWh** — le mix français.
Le même calcul, exécuté sur un mix nord-américain (ordre de grandeur 300 à
400 g/kWh selon les inventaires), émettrait **5 à 7 ×** plus.

Autrement dit : **déplacer le calcul change son empreinte plus que changer de
modèle**, sauf dans le cas extrême de TabPFN. C'est une limite structurelle de
l'indicateur — un chiffre CodeCarbon n'est comparable qu'à un autre chiffre
produit sur la même infrastructure. À citer avant toute comparaison entre
projets.

## Ce que la mesure ne vaut pas

**Sous ~2 secondes, la mesure n'est pas reproductible.** En comparant les deux
runs sur des modèles identiques :

| Modèle | Durée | Écart entre les deux runs |
|---|---|---|
| GradientBoosting | 0,5 s | **+39 %** |
| HistGradientBoosting | 0,6–1,8 s | **−58 %** |
| RandomForest | 1,3 s | −13 % |
| CatBoost | 3,7 s | −0,04 % |
| TabPFN | 36–38 s | −4 % |

Au-delà de 3 secondes la mesure est stable à quelques pourcents ; en dessous de
2 secondes elle est dominée par l'échantillonnage du wattmètre logiciel. Les
lignes GradientBoosting et RandomForest du tableau principal ne doivent donc
pas être comparées entre elles — seuls leurs ordres de grandeur tiennent.

**La première tâche du tracker absorbe son initialisation.** `energie-Dummy`,
d'une durée nulle, se voit attribuer 0,083 Wh — cinq fois plus que
GradientBoosting qui a réellement entraîné pendant une demi-seconde. La même
tâche placée en septième position ne mesure que 0,0001 Wh. Il faut donc soit
ignorer la première tâche, soit insérer une tâche de chauffe.

**La mesure est celle de la machine entière**, pas du processus Python : elle
inclut ce qui tourne à côté. Le mode `process` existe mais reste moins fiable
sur Windows.

## Ce qui reste à instrumenter

L'inférence n'a pas été mesurée. C'est pourtant elle qui tourne en continu, et
la seule question d'empreinte qui compte à terme est le **coût par prédiction
servie × volume servi**. Une piste naturelle de suite du projet : brancher le
même `start_task` sur l'appel de prédiction du Space et le rapporter au nombre
de bâtiments traités.

---

*Reproductible : section 4 de `notebooks/03_modelisation.ipynb`. Données brutes :
`emissions.csv` (agrégat par run) et `emissions_base_<run_id>.csv` (détail par
tâche).*
