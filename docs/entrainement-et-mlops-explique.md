# Entraînement, résultats et MLOps : tout comprendre

Projet `olist-lakehouse`. Ce document explique l'étape d'entraînement (`src/olist/train.py` et `src/jobs/05_train`), les résultats obtenus, et ce qui se passe ensuite en production : scoring, monitoring, dérive, réentraînement.

À placer dans `docs/` du dépôt.

---

## 0. L'essentiel en une page

**Ce qu'on prédit.** Au moment où une commande est approuvée, la probabilité qu'elle soit livrée après la date promise au client.

**Ce qu'on a fait.** Quatre modèles ont été entraînés sur les commandes de janvier 2017 à mars 2018, comparés sur avril-mai 2018 (validation), et le meilleur a été mesuré une seule fois sur juin-août 2018 (test).

**Ce qu'on a obtenu.**

| | Validation | Test |
|---|---|---|
| Meilleur modèle | régression logistique | (le même) |
| PR-AUC | 0,175 | 0,078 |
| Hasard (part de retards) | 0,055 | 0,036 |
| Lift (fois mieux que le hasard) | ×3,2 | ×2,1 |

**Ce que ça veut dire.** Le modèle apprend un vrai signal : il classe les retards deux à trois fois mieux que le hasard. Mais il se dégrade sur la période de test, parce que le monde a changé entre l'entraînement et le test. C'est un résultat modeste et honnête, et sa dégradation est exactement le phénomène que le MLOps sert à détecter et à corriger.

**Où on en est.** Le modèle est enregistré dans Unity Catalog, version 1, avec l'étiquette `champion`. Il reste à l'utiliser (scoring), à automatiser l'enchaînement (job), puis à surveiller (monitoring).

---

## 1. Le processus complet, du fichier CSV à la prédiction

```
9 fichiers CSV Olist
      │  upload
      ▼
Volume  olist_dev.raw.source_olist
      │  01_bronze   copie fidèle, tout en texte
      ▼
Bronze  lakehouse.bronze_*            9 tables
      │  02_silver   types, nettoyage, indicateurs de qualité, cible en_retard
      ▼
Silver  lakehouse.silver_*            8 tables
      │  03_gold     schéma en étoile
      ▼
Gold    dim_* et fct_*                5 dimensions, 2 tables de faits
      │  04_features une ligne par commande, 34 colonnes connues à l'approbation
      ▼
ml.features_commandes                 99 441 lignes
      │  05_train    4 modèles, comparaison, seuil, test
      ▼
MLflow (4 fiches d'entraînement)  ──►  Unity Catalog : delivery_delay_model v1 « champion »
      │  06_score    (à faire)
      ▼
gold_delivery_predictions             une probabilité par commande
      │  monitoring  (à faire)
      ▼
table de suivi : dérive, taux d'alertes, lift
```

| Étape | Notebook | Module | État |
|---|---|---|---|
| Ingestion | `01_bronze` | `bronze.py` | fait |
| Nettoyage | `02_silver` | `silver.py` | fait |
| Modèle en étoile | `03_gold` | `gold.py`, `geo.py` | fait |
| Features | `04_features` | `features.py` | fait |
| Entraînement | `05_train` | `train.py` | fait |
| Scoring | `06_score` | à écrire | à faire |
| Job qui enchaîne tout | Lakeflow | | à faire |
| Monitoring, réentraînement mensuel | | | à faire (temps 2) |

---

## 2. Ce qu'on prédit, et sur quelles commandes

### La cible

`en_retard = 1` si la date de livraison réelle est après la date promise, sinon `0`. Elle n'existe que pour une commande **livrée**. Une commande annulée ou encore en route a `en_retard = null`.

### La population d'entraînement

La table `features_commandes` garde les 99 441 commandes. La colonne `dans_population` vaut `true` pour celles qu'on a le droit d'utiliser pour apprendre :

- livrée (cible connue) ;
- approuvée, avec des dates cohérentes ;
- achetée entre le 1er janvier 2017 et le 26 août 2018.

Les commandes annulées ne sont pas supprimées de la table : elles sont marquées `false` et écartées au moment de l'entraînement. On garde toutes les lignes parce que le scoring, lui, devra calculer les features de commandes pas encore livrées.

### Le découpage dans le temps

| Période | Dates d'achat | Commandes | Retards | Taux |
|---|---|---|---|---|
| entraînement | janv. 2017 → mars 2018 | 64 020 | 5 110 | 7,98 % |
| validation | avr. → mai 2018 | 13 547 | 749 | 5,53 % |
| test | juin → 26 août 2018 | 18 487 | 672 | 3,63 % |

On découpe par date, pas au hasard, parce que c'est ce qui se passe en vrai : on apprend sur le passé pour prédire le futur. Un découpage au hasard mélangerait des commandes de mars 2018 dans l'entraînement et dans le test, et le modèle aurait l'air bien meilleur qu'il ne l'est.

Le rôle de chaque période :

- **entraînement** : le modèle apprend ;
- **validation** : on choisit le meilleur modèle et le seuil ;
- **test** : on note le résultat final, une seule fois, sans plus rien régler.

---

## 3. Les 34 features

Règle unique : **chaque colonne doit être connue au moment de l'approbation**. Tout ce qui arrive après (date de livraison, statut final, note du client) est interdit, et un contrôle automatique vérifie qu'aucune de ces colonnes n'est dans la table.

### Features de la commande (connues directement)

| Feature | Sens | Exemple |
|---|---|---|
| `delai_promis_j` | nombre de jours entre l'achat et la date promise | 21 |
| `delai_vendeur_j` | délai laissé au vendeur pour remettre le colis au transporteur | 5,9 |
| `delai_approbation_h` | heures entre l'achat et l'approbation du paiement | 0,18 (11 minutes) |
| `nb_articles`, `nb_produits`, `nb_vendeurs` | taille de la commande | 3, 2, 2 |
| `montant_articles`, `frais_port` | montants en réaux | 220, 50 |
| `ratio_frais_prix` | frais de port ÷ montant | 0,23 |
| `poids_total_g`, `volume_total_cm3` | taille physique du colis | 2 841, 23 920 |
| `nb_paiements`, `nb_echeances` | façon de payer | 1, 4 |
| `type_paiement_principal` | catégorie | `credit_card`, `boleto` |
| `categorie_principale` | catégorie de l'article le plus cher | `housewares` |

### Features géographiques

| Feature | Sens | Exemple |
|---|---|---|
| `distance_max_km` | distance vendeur → client (la plus grande s'il y a plusieurs vendeurs) | 1 457 |
| `meme_etat` | 1 si vendeur et client sont dans le même État | 0 |
| `etat_vendeur`, `etat_client` | États brésiliens | `RJ`, `BA` |

### Features de calendrier

`jour_semaine`, `mois`, `est_weekend`, `est_ferie`, `est_black_friday` : la date d'achat, vue sous plusieurs angles.

### Features d'historique (les plus délicates)

Elles résument le passé, **arrêté à la seconde de l'approbation**.

| Feature | Question posée, à l'instant de l'approbation |
|---|---|
| `vendeur_nb`, `vendeur_taux_retard` | combien de commandes de ce vendeur sont déjà livrées, et quelle part en retard ? |
| `vendeur_remise_nb`, `vendeur_remise_taux_retard` | combien de fois ce vendeur a déjà remis un colis, et quelle part hors délai ? |
| `route_nb`, `route_taux_retard` | sur cette route (État vendeur → État client), quel taux de retard jusqu'ici ? |
| `global_14j_nb`, `global_14j_taux_retard` | sur toute la plateforme, quel taux de retard parmi les colis arrivés ces 14 derniers jours ? |
| `global_30j_nb`, `global_30j_taux_retard` | pareil sur 30 jours |

**Exemple.** Le vendeur s2 a quatre commandes ; la commande o1 est approuvée le 12 février.

| Commande de s2 | Livrée le | En retard | Comptée pour o1 ? |
|---|---|---|---|
| A | 1er fév. | non | oui |
| B | 5 fév. | oui | oui |
| C | 20 fév. | oui | non : le 12 février, elle est encore en route |
| D | 28 fév. | non | non : future |

Pour o1 : `vendeur_nb = 2`, `vendeur_taux_retard = 0,50`.

Compter C ou D reviendrait à donner au modèle une information du futur. C'est la **fuite de données** : le modèle paraîtrait excellent en test et serait mauvais en production.

**Un vendeur sans historique** a `vendeur_nb = 0` et un taux `null` : inconnu, pas 0 %. Cela concerne 5,1 % des commandes.

**La preuve que c'est juste.** Le notebook `04_features` recalcule l'historique d'une commande par un simple filtre sur les dates et compare au résultat de la fonction : 72 livraisons passées et un taux de 0,0139 des deux côtés.

**Le « thermomètre » `global_14j`.** Il mesure l'état de la plateforme. Il a un défaut inévitable : un colis n'est compté qu'à son arrivée, donc le thermomètre réagit avec une à trois semaines de décalage. Ce défaut joue un rôle dans les résultats du test (section 7).

---

## 4. Le code d'entraînement : `src/olist/train.py`

Le fichier ne contient que des fonctions. C'est le notebook qui les appelle.

### `charger_jeux` : préparer les données

```python
df = (spark.table(table).filter("dans_population").select(...).toPandas())
return {p: d.reset_index(drop=True) for p, d in df.groupby("periode")}
```

1. Garde la population d'entraînement.
2. Fixe les types dans Spark : nombres en `double`, booléens en `0/1`, catégories vides remplacées par `"inconnu"`.
3. Convertit en pandas (96 000 lignes tiennent en mémoire, et scikit-learn travaille en pandas).
4. Renvoie trois tableaux : entraînement, validation, test.

### `preprocesseur` : transformer les colonnes en nombres

Un modèle ne lit que des nombres.

- **Catégories** : une colonne par valeur (one-hot). `etat_client = BA` devient `etat_client_BA = 1` et toutes les autres à 0. Une valeur vue moins de 50 fois est rangée dans « rare », et une valeur jamais vue ne fait pas planter le modèle.
- **Nombres, pour la régression logistique et la forêt** : une valeur manquante est remplacée par la médiane de l'entraînement, et une colonne « cette valeur manquait » est ajoutée. Puis tout est mis à la même échelle.
- **Nombres, pour LightGBM et XGBoost** : laissés tels quels. Ces deux modèles savent gérer les valeurs manquantes eux-mêmes, en apprenant de quel côté de chaque division les envoyer.

Tout ce qui est calculé ici (médianes, échelles, listes de catégories) l'est **sur l'entraînement seulement**, puis appliqué tel quel aux autres périodes.

### `modeles_candidats` : les quatre concurrents

| Modèle | Comment il décide |
|---|---|
| `logreg` (régression logistique) | additionne les features, chacune avec un poids, et transforme le total en probabilité |
| `random_forest` | 300 arbres de décision indépendants qui votent |
| `lightgbm` | 400 arbres construits l'un après l'autre, chacun corrige les erreurs des précédents |
| `xgboost` | même principe que LightGBM, autre implémentation |

Aucun modèle ne repondère les classes. Repondérer gonfle artificiellement les probabilités ; on préfère des probabilités fidèles et un seuil choisi après.

### `construire_pipeline` : coller préparation et modèle

Le pipeline est un seul objet qui contient la préparation **et** le modèle. On lui donne les 34 colonnes brutes, il fait le reste. Au scoring, il est impossible d'oublier une étape de préparation.

### `metriques`, `choisir_seuil`, `metriques_au_seuil`

Voir la section 6.

---

## 5. Le notebook `05_train`, cellule par cellule

| Cellule | Ce qu'elle fait | Ce que tu as vu |
|---|---|---|
| 1 | installe LightGBM et XGBoost, redémarre Python | (rien) |
| 2 | lit le paramètre `catalog`, construit les noms : table, modèle, expérience MLflow | `olist_dev.ml.features_commandes`, `olist_dev.ml.delivery_delay_model` |
| 3 | importe le module `olist.train`, coupe l'enregistrement automatique, pointe MLflow vers Unity Catalog et vers l'expérience | (rien) |
| 4 | charge les trois jeux, vérifie 64 020 lignes et 34 colonnes | 64 020 / 13 547 / 18 487 |
| 5 | entraîne les 4 modèles ; pour chacun, crée une fiche MLflow avec paramètres, métriques de validation et le modèle lui-même | 4 lignes de PR-AUC |
| 6 | trie les résultats et désigne le meilleur sur la validation | `logreg` |
| 7 | recharge le meilleur depuis MLflow, choisit le seuil sur la validation, mesure le test une fois, ajoute ces chiffres à la fiche du gagnant | seuil 0,208 ; test PR-AUC 0,078 |
| 8 | publie le gagnant dans Unity Catalog, lui colle l'étiquette `champion` et les notes `seuil` et `algorithme` | `v1 → alias 'champion'` |

Deux points à retenir sur l'ordre :

- La cellule 6 choisit le modèle **sans regarder le test**.
- La cellule 7 fixe le seuil sur la validation **avant** de toucher au test.

---

## 6. Les métriques, avec un exemple

Dix commandes, dont deux vraiment en retard (A et C). Le modèle leur donne une probabilité ; on les trie.

| Rang | Commande | Probabilité | En retard ? |
|---|---|---|---|
| 1 | A | 0,90 | oui |
| 2 | B | 0,70 | non |
| 3 | C | 0,60 | oui |
| 4 à 10 | D … J | 0,40 à 0,05 | non |

### Seuil, précision, rappel

Le **seuil** est la probabilité au-dessus de laquelle on déclenche une alerte.

| Seuil | Alertes | Précision : part des alertes justifiées | Rappel : part des retards attrapés |
|---|---|---|---|
| 0,5 | A, B, C | 2 sur 3 = 67 % | 2 sur 2 = 100 % |
| 0,8 | A | 1 sur 1 = 100 % | 1 sur 2 = 50 % |

Un seuil haut donne des alertes fiables mais rate des retards. Un seuil bas attrape tout mais noie l'équipe sous les fausses alertes.

Le **F1** combine précision et rappel en un seul chiffre. `choisir_seuil` essaie tous les seuils et garde celui dont le F1 est le plus haut.

### PR-AUC : la note sans seuil

Elle mesure une seule chose : les vrais retards sont-ils en haut du classement ?

On descend la liste, et à chaque vrai retard on note la précision à cet endroit :

- A au rang 1 : 1 retard sur 1 commande vue, soit 1,00 ;
- C au rang 3 : 2 retards sur 3 commandes vues, soit 0,67.

PR-AUC = (1,00 + 0,67) ÷ 2 = **0,83**.

Un modèle parfait obtient 1. Un modèle qui classe au hasard obtient à peu près la part de retards dans les données : ici 2 sur 10, soit 0,2. Cette part s'appelle la **prévalence**.

**Plus la PR-AUC est grande, mieux c'est.**

### Lift : la note comparée au hasard

Lift = PR-AUC ÷ prévalence = 0,83 ÷ 0,2 = **×4,2**. Le modèle classe 4,2 fois mieux que le hasard.

On a besoin du lift parce que la PR-AUC dépend de la rareté des retards. Quand il y en a moins, la PR-AUC baisse pour tout le monde, même pour le hasard. Le lift retire cet effet : il reste comparable d'une période à l'autre.

| Lift | Lecture |
|---|---|
| ×1 | pas mieux que le hasard, inutile |
| ×2 | utile mais faible |
| ×3 à ×4 | bon |

### Les deux métriques secondaires

- **ROC-AUC** : on tire au hasard un retard et une commande à l'heure ; c'est la probabilité que le modèle note le retard plus haut. Le hasard vaut 0,5. Elle est trop flatteuse quand les retards sont rares, donc on ne s'en sert pas pour décider.
- **Brier** : l'écart moyen entre la probabilité annoncée et ce qui s'est passé. **Plus il est petit, mieux c'est.**

### Dans le test, d'où vient le « hasard » ?

Le test est fait de commandes passées, déjà livrées : leurs réponses existent. On les a seulement cachées pendant qu'on construisait le modèle, comme un corrigé d'examen. Le hasard du test se calcule donc avec ce corrigé : 672 retards ÷ 18 487 commandes = 0,0363.

En vraie production, pour les commandes d'aujourd'hui, ce corrigé n'existe pas encore. Voir la section 9.

---

## 7. Les résultats obtenus

### Comparaison des quatre modèles (validation)

| Modèle | PR-AUC | ROC-AUC | Brier | Lift |
|---|---|---|---|---|
| **logreg** | **0,175** | 0,782 | 0,0517 | **×3,2** |
| lightgbm | 0,148 | 0,735 | 0,0540 | ×2,7 |
| xgboost | 0,140 | 0,734 | 0,0540 | ×2,5 |
| random_forest | 0,133 | 0,717 | 0,0548 | ×2,4 |

Hasard : 0,055. Les quatre apprennent quelque chose, et la régression logistique gagne sur toutes les métriques.

### Le seuil et le test

| | Validation | Test |
|---|---|---|
| Seuil | 0,208 (choisi ici) | 0,208 (appliqué tel quel) |
| Alertes | environ 1 450, soit 11 % des commandes | 6 271, soit 34 % |
| Précision | 19,5 % | 7,3 % |
| Rappel | 37,7 % | 67,7 % |
| Retards détectés / ratés | environ 282 / 467 | 455 / 217 |

En clair, sur le test : une commande sur trois est signalée, une alerte sur quatorze est justifiée, et deux retards sur trois sont attrapés.

### Pourquoi le modèle simple bat les modèles complexes

D'habitude, LightGBM et XGBoost gagnent sur ce type de données. Ici non, et l'explication la plus probable tient en une phrase : **les modèles complexes ont appris des détails de la période d'entraînement qui ne se répètent pas ensuite.**

- **Le monde change entre les périodes.** L'entraînement contient la crise de mars 2018 (19 % de retards). Un modèle à arbres peut retenir des combinaisons très précises de ce moment, par exemple « mois 3, thermomètre élevé, telle route ». En avril-mai, ces combinaisons n'existent plus.
- **`mois` est une feature piège.** Avec vingt mois de données, chaque mois n'apparaît qu'une ou deux fois. « Mars est risqué » est un accident de 2018, pas une règle.
- **La régression logistique est limitée, et ici c'est une qualité.** Elle ne peut apprendre que des tendances simples : plus c'est loin, plus c'est risqué ; un vendeur souvent en retard le reste. Ces tendances tiennent dans le temps.
- **Les arbres n'ont pas été bridés.** 400 arbres sans arrêt anticipé, avec seulement 5 110 retards pour apprendre : le surapprentissage est vraisemblable.

Cette explication est une hypothèse. On la vérifiera en comparant, pour chaque modèle, la PR-AUC sur l'entraînement et sur la validation : un grand écart signe le surapprentissage.

### Pourquoi le test est moins bon que la validation

Le lift passe de ×3,2 à ×2,1. Comme le lift corrige déjà la rareté des retards, il s'agit d'une vraie perte de qualité, pas d'un simple effet mécanique.

Le signe le plus parlant est le nombre d'alertes : 11 % des commandes en validation, 34 % en test. Les probabilités ont globalement monté, alors que les retards réels ont baissé.

La cause probable est le thermomètre. En juin, `global_14j_taux_retard` affiche encore environ 10 % : ce sont les colis commandés fin mai, pendant la grève des camionneurs au Brésil, qui arrivent en retard. Les commandes passées en juin, elles, n'ont que 1,2 % de retard. Le modèle croit donc la plateforme en crise au moment où elle va bien. À confirmer en comparant la moyenne du thermomètre et la moyenne des probabilités entre validation et test.

### Ce qu'on en conclut

1. Le modèle est utile mais faible : deux fois mieux que le hasard sur des données jamais vues.
2. Un seuil fixe en probabilité est fragile quand les probabilités dérivent. Une alternative plus robuste : alerter sur les X % de commandes les plus risquées de chaque semaine.
3. La dégradation en test est exactement ce qu'un monitoring aurait signalé en production dès la première semaine.

---

## 8. MLflow et Unity Catalog : qui fait quoi

Une simple liste Python suffit pour choisir le meilleur modèle, et c'est ce que fait la variable `resultats`. Mais cette liste disparaît à la fermeture du notebook, avec les modèles.

### MLflow : le cahier de laboratoire

Chaque entraînement y laisse une fiche permanente, appelée **run**, qui contient :

- les **paramètres** : comment le modèle a été réglé ;
- les **métriques** : `val_pr_auc`, `val_lift_pr_auc`, et pour le gagnant `test_pr_auc`, `seuil`, etc. ;
- le **modèle lui-même** : le pipeline complet, rechargeable.

Toutes les fiches du projet sont regroupées dans une **expérience**, `olist_retard_livraison`, visible dans le menu Experiments. On peut y cocher les quatre runs et cliquer sur Compare.

À quoi ça sert : retrouver dans un mois les réglages d'un modèle, comparer un nouvel essai aux anciens, et permettre à un autre programme de recharger le modèle.

### Unity Catalog : l'étagère officielle

MLflow garde tous les essais, bons et mauvais. Unity Catalog ne garde que les modèles **publiés**, numérotés par **version**.

```
MLflow : tous les essais                Unity Catalog : modèles publiés
  logreg          0,175  ── publié ──►  olist_dev.ml.delivery_delay_model
  lightgbm        0,148                   v1   étiquette « champion »
  xgboost         0,140                        notes : seuil = 0,208, algorithme = logreg
  random_forest   0,133
```

Unity Catalog **stocke** le modèle, il ne l'exécute pas. L'exécution se fait dans un notebook ou un job, sur le calcul serverless.

### L'étiquette `champion`

Le scoring chargera toujours :

```python
mlflow.sklearn.load_model("models:/olist_dev.ml.delivery_delay_model@champion")
```

Il ne connaît pas le numéro de version. Pour mettre un nouveau modèle en service, on déplace l'étiquette de v1 à v2. Pour revenir en arrière, on la remet sur v1. Le code du scoring ne change jamais.

Le seuil est rangé comme note sur la version : chaque version voyage avec son propre seuil.

---

## 9. En production : comment ça vit

### Une précision sur ce projet

Le jeu de données Olist est figé : il s'arrête en 2018. Il n'y a donc pas de vraies nouvelles commandes chaque jour. Ce qui suit décrit le fonctionnement d'un système réel ; dans le projet, on pourra le simuler plus tard avec un « rejoueur » qui fait arriver les commandes jour par jour.

### Les trois rythmes

| Rythme | Ce qu'on fait | Ce qu'on ne fait pas |
|---|---|---|
| chaque jour | **scoring** : le champion donne une probabilité aux commandes nouvellement approuvées | ni entraînement, ni test |
| chaque semaine | **contrôle** : dérive des données, taux d'alertes, performance sur les commandes maintenant livrées | pas d'entraînement |
| chaque mois, ou en urgence | **réentraînement** : entraînement, validation, test, puis duel avec le champion | |

On ne réentraîne pas chaque jour : une journée ajoute quelques centaines de commandes à 96 000, les réponses n'arrivent que deux à trois semaines plus tard, et chaque nouvelle version doit être validée.

### Comment savoir si une prédiction d'aujourd'hui était juste

Le jour même, on ne peut pas. La réponse arrive à la livraison.

```
jour J          commande approuvée   → scorée : probabilité 0,31, alerte oui
J + 15 jours    commande livrée      → réponse connue : en_retard = 0 ou 1
lundi suivant   contrôle             → on rapproche prédiction et réponse
fin du mois     réentraînement       → la commande rejoint les données d'entraînement
```

Concrètement, la table des prédictions garde `order_id`, la date de scoring, la probabilité et la version du modèle. Plus tard, on la joint à `fct_orders` sur `order_id` pour récupérer `en_retard`, et on calcule PR-AUC, prévalence et lift sur les commandes scorées trois semaines plus tôt.

### Les nouvelles données rejoignent-elles l'entraînement ?

Oui, au réentraînement suivant. Une commande scorée n'est pas perdue : une fois livrée, elle a sa réponse et devient une donnée d'apprentissage.

Le pipeline le fait déjà en partie : `04_features` recalcule toute la table à chaque exécution. Une commande livrée depuis la dernière fois a maintenant `en_retard` rempli et `dans_population = true`.

Ce qui manque : les dates du découpage sont des constantes dans `features.py`. Pour réentraîner chaque mois, il faudra les passer en paramètres, afin que la fenêtre glisse.

| Version | Entraînement | Validation | Test |
|---|---|---|---|
| v1 | janv. 2017 → mars 2018 | avr. → mai 2018 | juin → août 2018 |
| v2, un mois plus tard | janv. 2017 → avr. 2018 | mai → juin 2018 | juil. → sept. 2018 |

### Le duel champion contre challenger

Au réentraînement, les quatre algorithmes sont réentraînés sur les données récentes. Le meilleur est le **challenger**. Il est comparé au champion **sur les mêmes commandes récentes**, que ni l'un ni l'autre n'a vues. S'il gagne, l'étiquette `champion` passe sur lui. Sinon, rien ne change.

On ne ressort pas les anciens scores des anciens modèles : ils ont été mesurés sur d'anciennes données. La référence est toujours le champion, mesuré sur les données les plus récentes.

### Comment savoir que le modèle ne donne plus de bons résultats

Trois signaux, du plus rapide au plus fiable.

| Signal | Mesure | Disponible | Règle d'alerte proposée |
|---|---|---|---|
| dérive des données | PSI de chaque feature (section 10) | tout de suite | PSI > 0,25 |
| dérive des prédictions | part de commandes en alerte | tout de suite | hors de la bande habituelle, par exemple 6 à 16 % pour une habitude de 11 % |
| performance réelle | lift par semaine | après environ 3 semaines | lift en baisse de plus de 30 % par rapport à la validation, deux semaines de suite |

Avec cette dernière règle, le seuil serait ×3,2 × 0,7 = ×2,2. La période de test, à ×2,1, l'aurait déclenchée. Et le taux d'alertes, passé de 11 % à 34 %, aurait sonné bien avant.

### Le piège de la boucle de rétroaction

Si une alerte pousse l'équipe à agir et que la commande arrive finalement à l'heure, le modèle verra plus tard une commande « à risque » livrée à temps. Il en conclura à tort que ces signaux ne sont pas dangereux. Il faut garder la trace des commandes sur lesquelles on a agi, pour les marquer ou les exclure au réentraînement.

---

## 10. La dérive des données et le PSI

### Ce que mesure le PSI

Le PSI (Population Stability Index) compare la répartition d'une colonne aujourd'hui à sa répartition à l'entraînement.

Exemple inventé, sur `distance_max_km`. Les pourcentages sont des parts de commandes ; chaque colonne fait 100 %.

| Tranche | Entraînement | Ce mois-ci |
|---|---|---|
| moins de 300 km | 40 % | 25 % |
| 300 à 1 000 km | 35 % | 35 % |
| plus de 1 000 km | 25 % | 40 % |

PSI = somme de (actuel − référence) × ln(actuel ÷ référence)
= (0,25 − 0,40) × ln(0,25 ÷ 0,40) + 0 + (0,40 − 0,25) × ln(0,40 ÷ 0,25)
= 0,07 + 0 + 0,07 = **0,14**

| PSI | Lecture |
|---|---|
| moins de 0,1 | stable |
| 0,1 à 0,25 | dérive modérée : on note et on surveille |
| plus de 0,25 | dérive forte : on enquête |

Ces seuils sont des conventions du métier, pas des lois.

### Sur quelles colonnes

Sur les 34 features, plus la probabilité prédite. Pour une colonne numérique, on découpe en dix tranches contenant chacune 10 % des commandes d'entraînement. Pour une catégorie, chaque valeur est une tranche. Le calcul est fait par un job et rangé dans une table ; on regarde d'abord la probabilité prédite, puis les features qui pèsent le plus dans le modèle.

### Une alerte PSI ne déclenche pas un réentraînement

Elle déclenche une enquête.

```
alerte PSI sur une colonne
   │
   ├─ c'est un bug (la colonne est vide, un fichier est arrivé cassé)
   │     → on répare le pipeline, on ne réentraîne surtout pas
   │
   └─ c'est un vrai changement du monde
         │
         ├─ la performance reste bonne  → rien, on surveille
         └─ la performance baisse       → on réentraîne
```

La dérive dit que le monde a changé. Seule la performance dit que le modèle ne suit plus.

### Le PSI est-il dans le code ?

Pas encore. Voici la fonction prévue pour le monitoring, testée : elle renvoie environ 0 pour deux échantillons de la même loi et 0,21 pour un échantillon décalé.

```python
import numpy as np

def psi(reference, actuel, nb_tranches=10):
    """Écart de répartition entre l'entraînement (reference) et la période observée (actuel)."""
    reference, actuel = np.asarray(reference, float), np.asarray(actuel, float)
    reference, actuel = reference[~np.isnan(reference)], actuel[~np.isnan(actuel)]
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, nb_tranches + 1)))
    bornes[0], bornes[-1] = -np.inf, np.inf
    p_ref = np.clip(np.histogram(reference, bornes)[0] / len(reference), 1e-6, None)
    p_act = np.clip(np.histogram(actuel, bornes)[0] / len(actuel), 1e-6, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))
```

---

## 11. Ce qui est dans le code, et ce qui ne l'est pas

| Élément | Dans le code ? |
|---|---|
| pipeline Bronze, Silver, Gold avec contrôles | oui |
| features point-in-time, contrôle anti-fuite | oui |
| découpage temporel, 4 modèles, comparaison | oui |
| suivi MLflow : paramètres, métriques, modèles | oui |
| seuil choisi sur la validation, test mesuré une fois | oui |
| modèle versionné dans Unity Catalog, étiquette `champion`, seuil en note | oui |
| scoring avec `@champion` vers une table de prédictions | non, étape suivante |
| job qui enchaîne les notebooks | non |
| monitoring : taux d'alertes, PSI, lift hebdomadaire | non |
| dates de découpage en paramètres, réentraînement mensuel | non |
| duel champion contre challenger automatisé | non |
| bundle et CI/CD GitHub Actions | non |

---

## 12. Quoi faire maintenant, dans l'ordre

1. **Commit & Push** de `train.py` et `05_train`, et de ce document dans `docs/`.
2. **Capture d'écran** de la page Compare de MLflow, pour le README.
3. **Scoring** : notebook `06_score` qui charge `@champion`, lit son seuil, et écrit `gold_delivery_predictions`.
4. **Job Lakeflow** : enchaîner 01 → 02 → 03 → 04 → 06, avec le paramètre `catalog`.
5. **README** : schéma d'architecture, tableau de résultats, capture MLflow, limites connues.
6. **Candidature Inetum**, dès que les points 3 à 5 sont faits.
7. **Une expérience v2**, pas dix : retirer `mois`, brider les arbres, comparer avec et sans le thermomètre, le tout sur la validation. Si une variante gagne, elle devient v2 et prend l'étiquette `champion`.
8. **Bundle et GitHub Actions**, puis monitoring.

---

## 13. Les limites à écrire dans le README

- Le modèle ne prédit pas les annulations : il n'apprend que sur les commandes finalement livrées.
- La performance est modeste (lift ×2,1 en test) et baisse entre validation et test, à cause d'un changement de distribution.
- Le thermomètre global réagit avec retard et a probablement gonflé les probabilités en juin 2018.
- Le seuil fixe a produit trois fois plus d'alertes en test qu'en validation.
- Monitoring et réentraînement périodique sont décrits mais pas encore codés.

## 14. Quatre phrases pour l'entretien

- « Mes features historiques sont calculées à l'instant de l'approbation, et un test recalcule l'historique d'une commande par un autre chemin pour prouver qu'il n'y a pas de fuite. »
- « J'ai découpé dans le temps, choisi le modèle et le seuil sur la validation, et regardé le test une seule fois. »
- « La régression logistique a battu les boosters : avec un changement de distribution, le modèle simple généralise mieux. »
- « Le lift est passé de ×3,2 à ×2,1 et le taux d'alertes de 11 % à 34 % : c'est le cas d'école que le monitoring doit détecter, et la raison d'être du schéma champion contre challenger. »
