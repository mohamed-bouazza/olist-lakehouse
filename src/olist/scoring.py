"""Scoring : appliquer le modèle champion aux commandes et produire les prédictions."""
import pandas as pd
from pyspark.sql import SparkSession

from olist.train import COLONNES_X, colonnes_typees

ALIAS_CHAMPION = "champion"


def commandes_a_scorer(spark: SparkSession, table_features: str) -> pd.DataFrame:
    """Toutes les commandes approuvées, avec les 34 colonnes typées comme à l'entraînement."""
    return (
        spark.table(table_features)
        .filter("order_approved_at IS NOT NULL")
        .select("order_id", "order_approved_at", "periode", *colonnes_typees())
        .toPandas()
    )


def predire(pipeline, commandes: pd.DataFrame, seuil: float) -> pd.DataFrame:
    """Une probabilité et une décision d'alerte par commande."""
    proba = pipeline.predict_proba(commandes[COLONNES_X])[:, 1]
    return pd.DataFrame({
        "order_id": commandes["order_id"],
        "order_approved_at": commandes["order_approved_at"],
        "periode": commandes["periode"],
        "proba_retard": proba.astype(float),
        "alerte": proba >= seuil,
    })