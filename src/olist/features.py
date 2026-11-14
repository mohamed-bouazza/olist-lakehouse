"""Feature table : une ligne par commande, uniquement des informations connues à l'approbation."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SECONDES_PAR_JOUR = 86_400

# Population et découpage temporel (rapport d'EDA, §6.2)
DEBUT_MODELE = "2017-01-01"
FIN_ENTRAINEMENT = "2018-04-01"
FIN_VALIDATION = "2018-06-01"
FIN_MODELE = "2018-08-27"

# Colonnes connues APRÈS l'approbation : ne doivent jamais apparaître dans les features
COLONNES_INTERDITES = [
    "order_status", "order_delivered_carrier_date", "order_delivered_customer_date",
    "delai_livraison_j", "retard_j", "vendeur_en_retard", "note_avis",
]

FEATURES_NUMERIQUES = [
    "delai_promis_j", "delai_vendeur_j", "delai_approbation_h",
    "nb_articles", "nb_produits", "nb_vendeurs", "montant_articles", "frais_port", "ratio_frais_prix",
    "poids_total_g", "volume_total_cm3", "distance_max_km", "meme_etat",
    "nb_paiements", "nb_echeances", "jour_semaine", "mois",
    "vendeur_nb", "vendeur_taux_retard", "vendeur_remise_nb", "vendeur_remise_taux_retard",
    "route_nb", "route_taux_retard",
    "global_14j_nb", "global_14j_taux_retard", "global_30j_nb", "global_30j_taux_retard",
]
FEATURES_BOOLEENNES = ["est_weekend", "est_ferie", "est_black_friday"]
FEATURES_CATEGORIELLES = ["type_paiement_principal", "etat_client", "etat_vendeur", "categorie_principale"]


def historique_point_in_time(requetes: DataFrame, evenements: DataFrame, cles: list[str],
                             nom: str, fenetre_jours: int | None = None) -> DataFrame:
    """Pour chaque requête, compte les événements de mêmes clés survenus STRICTEMENT avant elle.

    requetes   : order_id, t (secondes), *cles
    evenements : t (secondes), *cles, retard (0/1)
    Renvoie    : order_id, {nom}_nb, {nom}_taux_retard
    """
    q = requetes.select(
        "order_id", F.col("t").alias("ts"), *cles,
        F.lit(0).alias("est_evenement"), F.lit(0).alias("retard"),
    )
    e = evenements.select(
        F.lit(None).cast("string").alias("order_id"), F.col("t").alias("ts"), *cles,
        F.lit(1).alias("est_evenement"), F.col("retard").cast("int").alias("retard"),
    )
    debut = Window.unboundedPreceding if fenetre_jours is None else -fenetre_jours * SECONDES_PAR_JOUR
    partition = Window.partitionBy(*cles) if cles else Window.partitionBy(F.lit(1))
    fenetre = partition.orderBy("ts").rangeBetween(debut, -1)   # -1 : strictement avant

    return (
        q.unionByName(e)
        .withColumn("nb", F.sum("est_evenement").over(fenetre))
        .withColumn("retards", F.sum("retard").over(fenetre))
        .filter(F.col("est_evenement") == 0)
        .select(
            "order_id",
            F.coalesce("nb", F.lit(0)).alias(f"{nom}_nb"),
            F.try_divide(F.col("retards"), F.col("nb")).alias(f"{nom}_taux_retard"),
        )
    )


def construire_features(fct_orders: DataFrame, fct_items: DataFrame, dim_customer: DataFrame,
                        dim_seller: DataFrame, dim_product: DataFrame, dim_date: DataFrame) -> DataFrame:
    # Vendeur et catégorie principaux : ceux de l'article le plus cher
    principal = (
        fct_items
        .join(dim_product.select("product_id", "product_category_name_english"), "product_id", "left")
        .groupBy("order_id")
        .agg(
            F.max_by("seller_id", "price").alias("vendeur_principal"),
            F.max_by("product_category_name_english", "price").alias("categorie_principale"),
        )
    )
    vendeurs = dim_seller.select(F.col("seller_id").alias("vendeur_principal"),
                                 F.col("seller_state").alias("etat_vendeur"))
    clients = dim_customer.select("customer_id", F.col("customer_state").alias("etat_client"))
    calendrier = dim_date.select(F.col("date_key").alias("date_key_achat"), "jour_semaine", "mois",
                                 "est_weekend", "est_ferie", "est_black_friday")

    base = (
        fct_orders
        .join(principal, "order_id", "left")
        .join(vendeurs, "vendeur_principal", "left")
        .join(clients, "customer_id", "left")
        .join(calendrier, "date_key_achat", "left")
        .withColumn("t_approbation", F.unix_timestamp("order_approved_at"))
    )

    # Événements passés : le résultat d'une commande est connu à la date où il se produit
    fiables = base.filter(F.col("en_retard").isNotNull()
                          & ~F.col("livree_avant_transporteur") & ~F.col("statut_incoherent"))
    livraisons = fiables.select(
        F.unix_timestamp("order_delivered_customer_date").alias("t"),
        "vendeur_principal", "etat_vendeur", "etat_client", "nb_vendeurs",
        F.col("en_retard").alias("retard"),
    )
    remises = (
        fiables
        .filter((F.col("nb_vendeurs") == 1) & F.col("vendeur_en_retard").isNotNull()
                & ~F.col("transporteur_avant_approbation"))
        .select(F.unix_timestamp("order_delivered_carrier_date").alias("t"),
                "vendeur_principal", F.col("vendeur_en_retard").alias("retard"))
    )
    requetes = (
        base.filter(F.col("t_approbation").isNotNull())
        .select("order_id", F.col("t_approbation").alias("t"),
                "vendeur_principal", "etat_vendeur", "etat_client")
    )

    historiques = [
        historique_point_in_time(requetes, livraisons.filter("nb_vendeurs = 1"),
                                 ["vendeur_principal"], "vendeur"),
        historique_point_in_time(requetes, remises, ["vendeur_principal"], "vendeur_remise"),
        historique_point_in_time(requetes, livraisons, ["etat_vendeur", "etat_client"], "route"),
        historique_point_in_time(requetes, livraisons, [], "global_14j", fenetre_jours=14),
        historique_point_in_time(requetes, livraisons, [], "global_30j", fenetre_jours=30),
    ]
    for h in historiques:
        base = base.join(h, "order_id", "left")

    achat = F.col("order_purchase_timestamp")
    return base.select(
        # Identifiants et cible (pas des features)
        "order_id", "vendeur_principal", "order_purchase_timestamp", "order_approved_at", "en_retard",
        (F.col("en_retard").isNotNull() & ~F.col("approbation_manquante")
         & ~F.col("livree_avant_transporteur") & ~F.col("statut_incoherent")
         & (achat >= DEBUT_MODELE) & (achat < FIN_MODELE)).alias("dans_population"),
        F.when(achat < FIN_ENTRAINEMENT, "entrainement")
         .when(achat < FIN_VALIDATION, "validation")
         .otherwise("test").alias("periode"),
        # Features de la commande
        "delai_promis_j", "delai_vendeur_j",
        ((F.col("t_approbation") - F.unix_timestamp("order_purchase_timestamp")) / 3600).alias("delai_approbation_h"),
        "nb_articles", "nb_produits", "nb_vendeurs",
        F.col("montant_articles").cast("double").alias("montant_articles"),
        F.col("frais_port").cast("double").alias("frais_port"),
        F.try_divide(F.col("frais_port"), F.col("montant_articles")).cast("double").alias("ratio_frais_prix"),
        "poids_total_g", "volume_total_cm3", "distance_max_km", "meme_etat",
        "nb_paiements", "nb_echeances", "type_paiement_principal",
        "etat_client", "etat_vendeur", "categorie_principale",
        "jour_semaine", "mois", "est_weekend", "est_ferie", "est_black_friday",
        # Features historiques (point-in-time)
        "vendeur_nb", "vendeur_taux_retard", "vendeur_remise_nb", "vendeur_remise_taux_retard",
        "route_nb", "route_taux_retard",
        "global_14j_nb", "global_14j_taux_retard", "global_30j_nb", "global_30j_taux_retard",
    )