"""Couche Bronze : copie fidèle des fichiers sources Olist, tout en texte."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Nom de la table Bronze -> nom du fichier dans le Volume source
FICHIERS_SOURCE = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Nombre de lignes de chaque source, mesuré pendant l'EDA
LIGNES_ATTENDUES = {
    "orders": 99_441,
    "order_items": 112_650,
    "payments": 103_886,
    "reviews": 99_224,
    "customers": 99_441,
    "sellers": 3_095,
    "products": 32_951,
    "geolocation": 1_000_163,
    "category_translation": 71,
}


def lire_csv_brut(spark: SparkSession, chemin: str) -> DataFrame:
    """Lit un CSV sans interpréter les types, et ajoute les colonnes de traçabilité."""
    return (
        spark.read
        .option("header", True)          # la première ligne contient les noms de colonnes
        .option("multiLine", True)       # des avis contiennent des retours à la ligne
        .option("quote", '"')
        .option("escape", '"')           # un guillemet dans le texte est doublé : ""
        .option("inferSchema", False)    # tout reste en texte : rien n'est perdu
        .option("encoding", "UTF-8")
        .csv(chemin)
        .select(
            "*",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_ingested_at"),
        )
    )


def ecrire_table(df: DataFrame, nom_complet: str) -> None:
    """Remplace entièrement le contenu de la table (idempotent)."""
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(nom_complet)
    )


def controler_table(spark: SparkSession, nom_complet: str, attendu: int) -> list[str]:
    """Renvoie la liste des problèmes de la table Bronze (vide si tout va bien)."""
    table = spark.table(nom_complet)
    problemes = []

    reel = table.count()
    if reel != attendu:
        problemes.append(f"{reel:,} lignes au lieu de {attendu:,}")

    non_texte = [c for c, t in table.dtypes if not c.startswith("_") and t != "string"]
    if non_texte:
        problemes.append(f"colonnes non textuelles : {non_texte}")

    sans_source = table.filter(F.col("_source_file").isNull()).count()
    if sans_source:
        problemes.append(f"{sans_source:,} lignes sans fichier source")

    return problemes