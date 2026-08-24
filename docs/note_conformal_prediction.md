# La prédiction conforme (MAPIE) — note technique

## Le problème

Un modèle de régression rend un nombre : *ce bâtiment consommera 2 547 969 kBtu*.
Il ne dit rien de sa fiabilité. Or sur le jeu Seattle, l'erreur relative médiane
est de 35 % mais le 90ᵉ centile dépasse 100 % — un bâtiment sur dix est faux d'un
facteur 2. **Le même nombre affiché recouvre des situations radicalement
différentes**, et l'utilisateur n'a aucun moyen de faire la différence.

Les approches classiques pour y remédier supposent quelque chose sur les erreurs :
la régression linéaire suppose des résidus gaussiens homoscédastiques, le bootstrap
suppose que le rééchantillonnage reproduit la variabilité réelle. Sur un modèle à
arbres et une cible étalée sur cinq ordres de grandeur, ces hypothèses ne tiennent pas.

## L'idée

> Plutôt que de prédire une valeur, prédire un **intervalle** garanti de contenir
> la vraie valeur dans au moins X % des cas — sans aucune hypothèse sur la forme
> des erreurs.

C'est la prédiction conforme (*conformal prediction*). MAPIE
(*Model Agnostic Prediction Interval Estimator*) en est l'implémentation
scikit-learn de référence.

## Comment ça marche : quatre étapes

L'astuce est presque triviale une fois vue. Version *split conformal*, la plus simple.

**1. Entraîner** le modèle sur un jeu A, normalement.

**2. Mesurer les erreurs passées** sur un jeu B **que le modèle n'a jamais vu**.
Pour chaque bâtiment de B on calcule le *score de conformité* — ici l'erreur
absolue `|y_réel − y_prédit|`. Sur nos données, 331 bâtiments de calibration :

```
scores :  min 0,000   médiane 0,381   max 2,478      (en espace log)
```

**3. Prendre le quantile** correspondant au niveau visé. Pour 75 % de couverture,
le 75ᵉ centile des scores :

```
q = 0,705
```

**4. Encadrer** toute prédiction future par `± q` :

```
[ŷ − q ; ŷ + q]
```

C'est tout. Aucun réentraînement, aucune hypothèse de loi.

## Pourquoi ça marche

L'intuition tient en une phrase :

> *Si sur des données que je n'avais jamais vues, 75 % de mes erreurs étaient
> inférieures à q, alors en ajoutant ± q autour d'une nouvelle prédiction je
> capture la vraie valeur environ 75 % du temps.*

La seule hypothèse est l'**échangeabilité** : les nouvelles données proviennent de
la même distribution que le jeu de calibration. Pas de normalité, pas de forme
paramétrique, pas de modèle bien spécifié — le modèle peut même être mauvais,
l'intervalle sera juste plus large. C'est ce qu'on appelle une garantie
*distribution-free*.

Vérification sur nos données : couverture visée 75 %, **couverture observée
78,5 %** sur le jeu de test.

## Un exemple complet

Bâtiment médian du jeu de test, cible = consommation d'énergie :

| | |
|---|---|
| Prédiction (log) | 14,751 |
| Intervalle (log) | [14,046 ; 15,456] |
| **Prédiction (kBtu)** | **2 547 969** |
| **Intervalle (kBtu)** | **[1 259 458 ; 5 154 713]** |

Trois choses à remarquer.

**L'intervalle est large** — un facteur 2,0 vers le bas comme vers le haut. Ce
n'est pas un défaut de la méthode : c'est l'incertitude réelle du modèle, que
l'affichage d'un seul nombre masquait entièrement. La prédiction conforme ne rend
pas le modèle plus incertain, elle rend son incertitude visible.

**L'intervalle est asymétrique en unité d'origine** : −1,29 M en dessous, +2,61 M
au-dessus. C'est mécanique : la cible est modélisée en `log1p`, l'intervalle est
calculé en log (où il est parfaitement symétrique : ±0,705), puis re-exponentié.
Comme `exp` est monotone croissante, la garantie de couverture traverse la
transformation intacte. Un intervalle symétrique en kBtu serait faux — il pourrait
descendre sous zéro.

**En log, ±q devient un facteur multiplicatif** : `exp(0,705) = 2,02`. L'intervalle
se lit donc « entre la moitié et le double de la prédiction » — formulation bien
plus parlante pour un utilisateur que deux bornes absolues, et c'est précisément
ce qui a motivé le choix du niveau à 75 % plutôt qu'à 90 %.

## Ne pas confondre : erreur typique et niveau de confiance

Deux nombres circulent en permanence et ne mesurent pas la même chose.

**35 % — la taille d'une erreur typique** (le MedAPE). On calcule l'erreur
relative `|réel − prédit| / réel` pour chaque bâtiment, on trie, on prend celle du
milieu. Traduction : *pour la moitié des bâtiments, la prédiction tombe à moins de
35 % de la vérité.* C'est un **résultat mesuré**.

**75 % — la fréquence à laquelle l'intervalle a raison** (la couverture). Ce n'est
pas une erreur mais une fréquence, et ce n'est pas un résultat : c'est un
**paramètre qu'on choisit** (`confidence_level=0.75`). On vérifie ensuite qu'il est
tenu — ici 81,3 % observés.

Les deux sont deux points de lecture de **la même distribution d'erreurs** :

| | Erreur relative |
|---|---|
| p50 | **35 %** ← le MedAPE lit la distribution au milieu |
| p75 | **63 %** ← l'intervalle à 75 % se cale ici |
| p90 | 100 % |

D'où l'écart qui surprend :

```
prédiction         : 2 547 969 kBtu
"typiquement ±35%" : 1 650 000 – 3 440 000     <- vrai une fois sur deux
intervalle 75 %    : 1 259 000 – 5 155 000     <- vrai trois fois sur quatre
```

L'intervalle est plus large que l'erreur typique, et c'est mécanique : il ne décrit
pas le cas courant, il doit **englober aussi les cas défavorables**. Plus le niveau
de confiance monte, plus loin dans la queue il faut aller — d'où un intervalle qui
s'élargit vite : 2,03 à 75 %, 2,98 à 90 %, 3,97 à 95 %.

Les deux se citent ensemble et se complètent : *« l'estimation est typiquement
juste à ±35 %, et l'intervalle affiché contient la vraie valeur 3 fois sur 4 »*.
La première phrase donne la précision usuelle, la seconde la fiabilité garantie.

## Comment lire un graphique de couverture

Piège classique : trier les bâtiments par **valeur réelle**. L'intervalle étant
centré sur la **prédiction**, le ruban part alors dans tous les sens et semble
erratique — alors qu'il est simplement décalé par l'erreur du modèle.

Il faut trier par **prédiction**. Le ruban devient un bandeau lisse autour de la
ligne de prédiction, et la dispersion des points réels autour d'elle est
exactement l'erreur du modèle. Ce qui paraissait un défaut de la méthode se lit
alors pour ce que c'est.

## Ce que la garantie dit — et ne dit pas

**Elle est marginale, pas conditionnelle.** La couverture annoncée vaut *en moyenne
sur l'ensemble* des bâtiments. Rien ne la garantit sur chaque sous-groupe :
les hôpitaux peuvent être couverts à 85 % et les entrepôts à 68 %, la moyenne
restant à 75 %. C'est la limite principale, et il faut la citer honnêtement.

**Elle ne corrige pas un modèle biaisé.** Si le modèle sous-estime
systématiquement, l'intervalle sera décalé de la même façon — simplement assez
large pour contenir la vérité dans la proportion visée.

**Elle suppose l'échangeabilité.** Un bâtiment hors domaine — typologie jamais
vue, autre ville, autre climat — casse l'hypothèse, et la couverture n'est plus
garantie.

## Les variantes dans MAPIE

| Classe | Principe | Quand l'utiliser |
|---|---|---|
| `SplitConformalRegressor` | Un jeu de calibration dédié, mis de côté | Beaucoup de données ; le plus simple à comprendre |
| `CrossConformalRegressor` | Scores calculés **hors fold** en validation croisée | **Petits jeux** — aucune donnée sacrifiée, plus robuste |
| `ConformalizedQuantileRegressor` | S'appuie sur une régression quantile | Quand on veut des intervalles **de largeur variable** selon le bâtiment |

Sur le projet Seattle (1655 lignes), `CrossConformalRegressor` s'impose : le split
conformal aurait confisqué 20 % des données pour la seule calibration.

Note sur les deux premières : elles produisent des intervalles de **largeur
constante** (± q pour tout le monde). La troisième les fait varier — un bâtiment
atypique reçoit un intervalle plus large. C'est plus fin, au prix d'un modèle
quantile à entraîner.

## Le code

```python
from mapie.regression import CrossConformalRegressor

mapie = CrossConformalRegressor(
    estimator=GradientBoostingRegressor(),
    confidence_level=0.75,
    cv=5,
    method="plus",          # CV+
)

# fit + calcul des scores de conformité hors fold, en une passe
mapie.fit_conformalize(X_train, y_train)

y_pred, intervals = mapie.predict_interval(X_test)
lower, upper = intervals[:, 0, 0], intervals[:, 1, 0]

# cible en log -> on repasse en unité métier à la toute fin
y_pred, lower, upper = np.expm1(y_pred), np.expm1(lower), np.expm1(upper)

couverture = ((y_test >= lower) & (y_test <= upper)).mean()
```

**Piège à éviter** : ne pas envelopper le modèle dans un `TransformedTargetRegressor`
qui ferait la transformation `log1p`/`expm1` en interne. MAPIE calibrerait alors
sur des kBtu, les résidus seraient symétriques en unité brute, et la borne basse
pourrait devenir négative. Le modèle doit sortir du log, MAPIE calibrer en log, et
`expm1` s'appliquer en dernier — au point comme aux deux bornes.

## À retenir

- Un intervalle **garanti** sans hypothèse de loi, pour le prix d'un jeu de
  calibration et d'un calcul de quantile.
- Fonctionne autour de **n'importe quel** modèle scikit-learn, sans le modifier.
- La garantie est **marginale** : vraie en moyenne, pas par sous-groupe.
- Un intervalle large n'est pas un échec — c'est une incertitude enfin visible.
- Sur cible en log : calibrer en log, re-exponentier à la fin, assumer l'asymétrie.
