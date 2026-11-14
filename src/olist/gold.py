"""Couche Gold : schéma en étoile pour l'analyse et le ML."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from olist.geo import distance_km

# Jours fériés nationaux brésiliens (dont carnaval et vendredi saint) sur la période des données
JOURS_FERIES = [
    "2016-01-01", "2016-02-08", "2016-02-09", "2016-03-25", "2016-04-21", "2016-05-01",
    "2016-09-07", "2016-10-12", "2016-11-02", "2016-11-15", "2016-12-25",
    "2017-01-01", "2017-02-27", "2017-02-28", "2017-04-14", "2017-04-21", "2017-05-01",
    "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
    "2018-01-01", "2018-02-12", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01",
    "2018-09-07", "2018-10-12", "2018-11-02", "2018-11-15", "2018-12-25",
]
BLACK_FRIDAY = ["2016-11-25", "2017-11-24", "2018-11-23"]


def cle_date(colonne: str):
    """Horodatage -> clé de date entière : 2017-10-02 10:56 -> 20171002."""
    return F.date_format(colonne, "yyyyMMdd").cast("int")


# ---------- Dimensions ----------

def dim_date(spark: SparkSession, debut: str = "2016-01-01", fin: str = "2018-12-31") -> DataFrame:
    jours = spark.sql(
        f"SELECT explode(sequence(to_date('{debut}'), to_date('{fin}'), interval 1 day)) AS date"
    )
    return jours.select(
        F.date_format("date", "yyyyMMdd").cast("int").alias("date_key"),
        "date",
        F.year("date").alias("annee"),
        F.month("date").alias("mois"),
        F.dayofmonth("date").alias("jour"),
        F.dayofweek("date").alias("jour_semaine"),          # 1 = dimanche … 7 = samedi
        F.weekofyear("date").alias("semaine"),
        F.dayofweek("date").isin(1, 7).alias("est_weekend"),
        F.date_format("date", "yyyy-MM-dd").isin(JOURS_FERIES).alias("est_ferie"),
        F.date_format("date", "yyyy-MM-dd").isin(BLACK_FRIDAY).alias("est_black_friday"),
    )


def dim_geography(silver_geolocation: DataFrame) -> DataFrame:
    return silver_geolocation.select(
        F.col("geolocation_zip_code_prefix").alias("zip_code_prefix"),
        F.col("geolocation_lat").alias("lat"),
        F.col("geolocation_lng").alias("lng"),
        F.col("geolocation_state").alias("geo_state"),
        "nb_points",
    )


def dim_customer(silver_customers: DataFrame, silver_geolocation: DataFrame) -> DataFrame:
    geo = dim_geography(silver_geolocation)
    return (
        silver_customers
        .join(geo, silver_customers.customer_zip_code_prefix == geo.zip_code_prefix, "left")
        .select(
            "customer_id", "customer_unique_id", "customer_zip_code_prefix",
            "customer_city", "customer_state",
            F.col("lat").alias("customer_lat"), F.col("lng").alias("customer_lng"),
        )
    )


def dim_seller(silver_sellers: DataFrame, silver_geolocation: DataFrame) -> DataFrame:
    geo = dim_geography(silver_geolocation)
    return (
        silver_sellers
        .join(geo, silver_sellers.seller_zip_code_prefix == geo.zip_code_prefix, "left")
        .select(
            "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
            F.col("lat").alias("seller_lat"), F.col("lng").alias("seller_lng"),
        )
    )


def dim_product(silver_products: DataFrame) -> DataFrame:
    return (
        silver_products.drop("_source_file")
        .withColumn("product_volume_cm3",
                    F.col("product_length_cm") * F.col("product_height_cm") * F.col("product_width_cm"))
    )


# ---------- Faits ----------

def fct_order_items(silver_order_items: DataFrame, silver_orders: DataFrame,
                    dim_cust: DataFrame, dim_sell: DataFrame) -> DataFrame:
    """Grain : un article d'une commande."""
    commandes = silver_orders.select("order_id", "customer_id", "order_purchase_timestamp")
    clients = dim_cust.select("customer_id", "customer_state", "customer_lat", "customer_lng")
    vendeurs = dim_sell.select("seller_id", "seller_state", "seller_lat", "seller_lng")
    return (
        silver_order_items
        .join(commandes, "order_id")
        .join(clients, "customer_id", "left")
        .join(vendeurs, "seller_id", "left")
        .select(
            "order_id", "order_item_id", "product_id", "seller_id", "customer_id",
            cle_date("order_purchase_timestamp").alias("date_key_achat"),
            "shipping_limit_date", "price", "freight_value",
            (F.col("seller_state") == F.col("customer_state")).alias("meme_etat"),
            distance_km("seller_lat", "seller_lng", "customer_lat", "customer_lng").alias("distance_km"),
        )
    )


def fct_orders(silver_orders: DataFrame, fct_items: DataFrame, dim_prod: DataFrame,
               silver_payments: DataFrame, silver_reviews: DataFrame) -> DataFrame:
    """Grain : une commande. Contient des colonnes d'analyse connues APRÈS l'approbation."""
    articles = (
        fct_items
        .join(dim_prod.select("product_id", "product_weight_g", "product_volume_cm3"), "product_id", "left")
        .groupBy("order_id")
        .agg(
            F.count("*").alias("nb_articles"),
            F.countDistinct("product_id").alias("nb_produits"),
            F.countDistinct("seller_id").alias("nb_vendeurs"),
            F.sum("price").alias("montant_articles"),
            F.sum("freight_value").alias("frais_port"),
            F.sum("product_weight_g").alias("poids_total_g"),
            F.sum("product_volume_cm3").alias("volume_total_cm3"),
            F.max("distance_km").alias("distance_max_km"),
            F.min(F.col("meme_etat").cast("int")).alias("meme_etat"),
            F.max("shipping_limit_date").alias("date_limite_expedition"),
        )
    )
    paiements = silver_payments.groupBy("order_id").agg(
        F.sum("payment_value").alias("montant_paye"),
        F.count("*").alias("nb_paiements"),
        F.max("payment_installments").alias("nb_echeances"),
        F.max_by("payment_type", "payment_value").alias("type_paiement_principal"),
    )
    avis = silver_reviews.groupBy("order_id").agg(
        F.max_by("review_score", "review_answer_timestamp").alias("note_avis"),
    )
    return (
        silver_orders.drop("_source_file")
        .join(articles, "order_id", "left")
        .join(paiements, "order_id", "left")
        .join(avis, "order_id", "left")
        .withColumn("date_key_achat", cle_date("order_purchase_timestamp"))
        # --- connues à l'approbation : candidates pour les features ---
        .withColumn("delai_promis_j",
                    F.datediff("order_estimated_delivery_date", "order_purchase_timestamp"))
        .withColumn("delai_vendeur_j",
                    (F.unix_timestamp("date_limite_expedition") - F.unix_timestamp("order_approved_at")) / 86400)
        # --- connues APRÈS l'approbation : analyse uniquement, jamais des features ---
        .withColumn("delai_livraison_j",
                    F.datediff("order_delivered_customer_date", "order_purchase_timestamp"))
        .withColumn("retard_j",
                    F.datediff("order_delivered_customer_date", "order_estimated_delivery_date"))
        .withColumn("vendeur_en_retard",
                    (F.col("order_delivered_carrier_date") > F.col("date_limite_expedition")).cast("int"))
    )