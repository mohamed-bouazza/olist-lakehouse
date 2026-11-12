"""Couche Silver : données typées, normalisées et contrôlées (règles issues de l'EDA)."""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

FORMAT_HORODATAGE = "yyyy-MM-dd HH:mm:ss"
LAT_BRESIL = (-33.75, 5.27)
LNG_BRESIL = (-73.99, -34.79)
ACCENTS = "áàâãäéèêëíìîïóòôõöúùûüç"
SANS_ACCENTS = "aaaaaeeeeiiiiooooouuuuc"

DATES_COMMANDE = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


# ---------- Conversions réutilisables ----------

def en_horodatage(nom: str) -> Column:
    """Texte -> horodatage ; une valeur illisible devient null au lieu de planter."""
    return F.try_to_timestamp(F.col(nom), F.lit(FORMAT_HORODATAGE)).alias(nom)


def en_nombre(nom: str, type_sql: str, nouveau_nom: str | None = None) -> Column:
    """Texte -> nombre (INT, DOUBLE, DECIMAL…) ; une valeur illisible devient null."""
    return F.expr(f"try_cast(`{nom}` AS {type_sql})").alias(nouveau_nom or nom)


def code_postal(nom: str) -> Column:
    """Code postal en texte sur 5 caractères : 1037 -> 01037."""
    return F.lpad(F.trim(F.col(nom)), 5, "0").alias(nom)


def texte_normalise(nom: str) -> Column:
    """Minuscules, sans accents, sans espaces en trop : ' São  Paulo ' -> 'sao paulo'."""
    texte = F.translate(F.lower(F.trim(F.col(nom))), ACCENTS, SANS_ACCENTS)
    return F.regexp_replace(texte, r"\s+", " ").alias(nom)


def majuscules(nom: str) -> Column:
    return F.upper(F.trim(F.col(nom))).alias(nom)


# ---------- Une fonction par table ----------

def silver_orders(bronze: DataFrame) -> DataFrame:
    livree = (F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNotNull()
    return (
        bronze.select(
            "order_id",
            "customer_id",
            F.lower(F.trim("order_status")).alias("order_status"),
            *[en_horodatage(c) for c in DATES_COMMANDE],
            "_source_file",
        )
        .withColumn("approbation_manquante", F.col("order_approved_at").isNull())
        .withColumn("transporteur_avant_approbation", F.coalesce(
            F.col("order_delivered_carrier_date") < F.col("order_approved_at"), F.lit(False)))
        .withColumn("livree_avant_transporteur", F.coalesce(
            F.col("order_delivered_customer_date") < F.col("order_delivered_carrier_date"), F.lit(False)))
        .withColumn("statut_incoherent", F.coalesce(
            ((F.col("order_status") == "delivered") & F.col("order_delivered_customer_date").isNull())
            | ((F.col("order_status") != "delivered") & F.col("order_delivered_customer_date").isNotNull()),
            F.lit(False)))
        .withColumn("en_retard", F.when(livree, (
            F.to_date("order_delivered_customer_date") > F.to_date("order_estimated_delivery_date")
        ).cast("int")))
    )


def silver_order_items(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        "order_id",
        en_nombre("order_item_id", "INT"),
        "product_id",
        "seller_id",
        en_horodatage("shipping_limit_date"),
        en_nombre("price", "DECIMAL(10,2)"),
        en_nombre("freight_value", "DECIMAL(10,2)"),
        "_source_file",
    )


def silver_payments(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        "order_id",
        en_nombre("payment_sequential", "INT"),
        F.lower(F.trim("payment_type")).alias("payment_type"),
        en_nombre("payment_installments", "INT"),
        en_nombre("payment_value", "DECIMAL(10,2)"),
        "_source_file",
    )


def silver_reviews(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        "review_id",
        "order_id",
        en_nombre("review_score", "INT"),
        "review_comment_title",
        "review_comment_message",
        en_horodatage("review_creation_date"),
        en_horodatage("review_answer_timestamp"),
        "_source_file",
    )


def silver_customers(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        "customer_id",
        "customer_unique_id",
        code_postal("customer_zip_code_prefix"),
        texte_normalise("customer_city"),
        majuscules("customer_state"),
        "_source_file",
    )


def silver_sellers(bronze: DataFrame) -> DataFrame:
    return bronze.select(
        "seller_id",
        code_postal("seller_zip_code_prefix"),
        texte_normalise("seller_city"),
        majuscules("seller_state"),
        "_source_file",
    )


def silver_products(bronze: DataFrame, traduction: DataFrame) -> DataFrame:
    trad = traduction.select(
        F.trim("product_category_name").alias("product_category_name"),
        F.trim("product_category_name_english").alias("categorie_en"),
    )
    return (
        bronze.select(
            "product_id",
            F.trim("product_category_name").alias("product_category_name"),
            en_nombre("product_name_lenght", "INT", "product_name_length"),
            en_nombre("product_description_lenght", "INT", "product_description_length"),
            en_nombre("product_photos_qty", "INT"),
            en_nombre("product_weight_g", "INT"),
            en_nombre("product_length_cm", "INT"),
            en_nombre("product_height_cm", "INT"),
            en_nombre("product_width_cm", "INT"),
            "_source_file",
        )
        .join(trad, "product_category_name", "left")
        .withColumn("product_category_name_english",
                    F.coalesce("categorie_en", "product_category_name", F.lit("unknown")))
        .withColumn("product_category_name", F.coalesce("product_category_name", F.lit("unknown")))
        .drop("categorie_en")
    )


def silver_geolocation(bronze: DataFrame) -> DataFrame:
    """Grain : un code postal (et non plus un point GPS)."""
    return (
        bronze.select(
            code_postal("geolocation_zip_code_prefix"),
            en_nombre("geolocation_lat", "DOUBLE"),
            en_nombre("geolocation_lng", "DOUBLE"),
            majuscules("geolocation_state"),
        )
        .filter(F.col("geolocation_lat").between(*LAT_BRESIL)
                & F.col("geolocation_lng").between(*LNG_BRESIL))
        .groupBy("geolocation_zip_code_prefix")
        .agg(
            F.percentile_approx("geolocation_lat", 0.5).alias("geolocation_lat"),
            F.percentile_approx("geolocation_lng", 0.5).alias("geolocation_lng"),
            F.mode("geolocation_state").alias("geolocation_state"),
            F.count("*").alias("nb_points"),
        )
    )


# ---------- Registre : table Silver -> (fonction, tables Bronze en entrée) ----------

TABLES_SILVER = {
    "orders": (silver_orders, ["orders"]),
    "order_items": (silver_order_items, ["order_items"]),
    "payments": (silver_payments, ["payments"]),
    "reviews": (silver_reviews, ["reviews"]),
    "customers": (silver_customers, ["customers"]),
    "sellers": (silver_sellers, ["sellers"]),
    "products": (silver_products, ["products", "category_translation"]),
    "geolocation": (silver_geolocation, ["geolocation"]),
}

CLES = {
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "reviews": ["review_id", "order_id"],
    "customers": ["customer_id"],
    "sellers": ["seller_id"],
    "products": ["product_id"],
    "geolocation": ["geolocation_zip_code_prefix"],
}


# ---------- Contrôles ----------

def problemes_de_cle(df: DataFrame, cle: list[str]) -> list[str]:
    problemes = []
    doublons = df.count() - df.select(*cle).distinct().count()
    if doublons:
        problemes.append(f"{doublons:,} doublons de clé {cle}")
    nulles = df.filter(" OR ".join(f"`{c}` IS NULL" for c in cle)).count()
    if nulles:
        problemes.append(f"{nulles:,} clés nulles")
    return problemes


def conversions_echouees(bronze: DataFrame, silver: DataFrame) -> dict[str, int]:
    """Colonnes où Silver a plus de valeurs nulles que Bronze : des conversions ont échoué."""
    communes = [c for c in silver.columns if c in bronze.columns and not c.startswith("_")]

    def nulls(df: DataFrame) -> dict[str, int]:
        ligne = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in communes]).first()
        return ligne.asDict()

    avant, apres = nulls(bronze), nulls(silver)
    return {c: apres[c] - avant[c] for c in communes if apres[c] > avant[c]}