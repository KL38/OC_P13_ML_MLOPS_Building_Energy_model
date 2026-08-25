# Le plafond de performance — pourquoi ±2× est la bonne réponse

**Destination : rapport, sections « limites » et « perspectives d'évolution ».**

Ce document établit que l'incertitude du modèle n'est pas un défaut de
modélisation mais une propriété du problème. C'est le résultat le plus solide du
projet, et celui qui doit être présenté avant qu'un lecteur ne conclue de lui-même
que le modèle est médiocre.

## Le constat qui dérange

Le modèle retenu produit des intervalles à 75 % d'un facteur **2,02** : la vraie
consommation se situe entre la moitié et le double de la prédiction.
Le bâtiment typique est estimé à ±35 %, et un bâtiment sur dix est faux d'un
facteur 2 ou plus.

Présenté brut, cela peut sembler faible. La question à trancher est donc : **est-ce le
modèle qui est mauvais, ou le problème qui est intrinsèquement incertain ?**

## La mesure

On regroupe les bâtiments que le modèle voit comme **interchangeables** — même
groupe d'usage dominant, même tranche de surface — et on observe la dispersion
réelle de leur intensité énergétique (kBtu par pied carré). Cet écart est
irréductible : aucun modèle utilisant ces variables ne peut le prédire.

| Groupe comparable | n | Intensité p10 | Intensité p90 | Ratio |
|---|---|---|---|---|
| Bureaux, grande tranche | 163 | 27,2 | 90,9 | **3,35×** |
| Bureaux, tranche moyenne | 120 | 31,4 | 105,3 | 3,35× |
| Entrepôts secs | 69 | 9,8 | 71,0 | **7,27×** |
| Salles / assemblée | 70 | 14,6 | 99,7 | 6,83× |
| Enseignement | 62 | 24,9 | 61,5 | 2,47× |

**163 immeubles de bureaux de taille comparable consomment entre 27 et 91 kBtu par
pied carré.** Le modèle ne dispose d'aucune variable pour les départager.

- Ratio p90/p10 médian sur l'ensemble des groupes : **4,10×**
- Écart-type intra-groupe du log d'intensité : **0,699**

## Le modèle est au plafond

Un intervalle **au même niveau de confiance**, construit sur cette seule dispersion
irréductible, vaudrait `exp(1,150 × 0,699)`, soit un facteur **2,24**.

L'intervalle effectivement produit par le modèle vaut **2,02**.

(La comparaison se fait à niveau de confiance égal. À 90 %, le plancher serait de
3,16 et le modèle produirait 2,98 — l'écart relatif est le même.)

> Le modèle est donc **plus serré que la dispersion naturelle entre bâtiments
> indiscernables**. Il extrait quasiment tout le signal que ces variables
> contiennent.

## Pourquoi des bâtiments identiques consomment si différemment

Parce que les variables disponibles décrivent **ce qu'un bâtiment est**, alors que
la consommation dépend de **comment il est utilisé et exploité**.

- **Le taux d'occupation et les horaires.** Un plateau de bureaux à 30 %
  d'occupation contre un autre plein, ouvert 40 h ou 90 h par semaine. Facteur 2
  à lui seul.
- **Les équipements installés.** Une salle serveurs au sous-sol, une cuisine
  professionnelle, de l'imagerie médicale. Invisible dans les données, décisif
  dans la facture.
- **L'âge et le rendement des systèmes.** Une chaudière de 1975 contre une pompe à
  chaleur récente, à bâtiment identique.
- **La qualité de l'enveloppe.** Isolation, vitrages, étanchéité à l'air — deux
  immeubles de 1960 dont l'un a été rénové et l'autre non sont indiscernables dans
  ce jeu de données.
- **Les consignes et la régulation.** 19 °C ou 23 °C, réduit de nuit ou pas.

Rien de tout cela ne figure dans le benchmark de Seattle, et rien ne se déduit
d'une surface et d'une année de construction.

## Ce qu'il faut en conclure

**Pour la section « limites ».** L'incertitude affichée n'est pas un aveu de
faiblesse, c'est une mesure honnête. Elle était présente dans les modèles de P3 —
simplement jamais quantifiée, puisqu'une estimation ponctuelle ne dit rien de sa
fiabilité.

**Pour la section « perspectives ».** Améliorer la précision ne passe pas par un
meilleur algorithme. Le benchmark le montre déjà : six familles de modèles se
tiennent en 4 points de R², et le modèle de fondation TabPFN ne gagne que 2 points
— dans le bruit d'échantillonnage. La marge n'est pas dans l'algorithme, elle est
dans les **données** :

1. Données d'occupation (effectifs, horaires d'ouverture)
2. Inventaire des équipements énergivores
3. Âge et type des systèmes de chauffage et de climatisation
4. Indicateurs d'enveloppe (année de rénovation, performance thermique)
5. Sous-comptage par usage

**Pour l'usage du modèle.** Une précision à ±2× reste exploitable pour **classer un
parc et prioriser des audits** — le cas d'usage visé. Elle ne l'est pas pour
facturer, contractualiser ou dimensionner un investissement. Cette frontière doit
être écrite dans l'interface, pas seulement dans le rapport.

---

*Reproductible : section 10 de `notebooks/03_modelisation.ipynb`.*
