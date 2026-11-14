"""Entraînement : jeux de données, modèles candidats, métriques et choix du seuil."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from olist.features import FEATURES_BOOLEENNES, FEATURES_CATEGORIELLES, FEATURES_NUMERIQUES

CIBLE = "en_retard"
COLONNES_X = FEATURES_NUMERIQUES + FEATURES_BOOLEENNES + FEATURES_CATEGORIELLES
GRAINE = 42
GERENT_NAN = {"lightgbm", "xgboost"}


def charger_jeux(spark: SparkSession, table: str) -> dict[str, pd.DataFrame]:
    """Population d'entraînement, typée proprement, découpée par période."""
    df = (
        spark.table(table)
        .filter("dans_population")
        .select(
            "order_id", "periode", F.col(CIBLE).cast("int").alias(CIBLE),
            *[F.col(c).cast("double").alias(c) for c in FEATURES_NUMERIQUES],
            *[F.col(c).cast("int").alias(c) for c in FEATURES_BOOLEENNES],
            *[F.coalesce(F.col(c), F.lit("inconnu")).alias(c) for c in FEATURES_CATEGORIELLES],
        )
        .toPandas()
    )
    return {p: d.reset_index(drop=True) for p, d in df.groupby("periode")}


def preprocesseur(imputer: bool = True) -> ColumnTransformer:
    """Prépare les 34 colonnes. imputer=False laisse les NaN aux boosters, qui les gèrent nativement."""
    if imputer:
        numerique = Pipeline([
            ("imputation", SimpleImputer(strategy="median", add_indicator=True)),
            ("echelle", StandardScaler()),
        ])
    else:
        numerique = "passthrough"
    categoriel = OneHotEncoder(handle_unknown="infrequent_if_exist",
                               min_frequency=50, sparse_output=False)
    return ColumnTransformer([
        ("num", numerique, FEATURES_NUMERIQUES + FEATURES_BOOLEENNES),
        ("cat", categoriel, FEATURES_CATEGORIELLES),
    ])


def modeles_candidats() -> dict:
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=GRAINE),
        "lightgbm": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=GRAINE, verbose=-1),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            random_state=GRAINE, n_jobs=-1),
    }


def construire_pipeline(nom: str, modele) -> Pipeline:
    return Pipeline([
        ("preparation", preprocesseur(imputer=nom not in GERENT_NAN)),
        ("modele", modele),
    ])


def metriques(y, proba) -> dict[str, float]:
    """Métriques indépendantes du seuil."""
    prevalence = float(np.mean(y))
    pr_auc = float(average_precision_score(y, proba))
    return {
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "prevalence": prevalence,
        "lift_pr_auc": pr_auc / prevalence,
    }


def choisir_seuil(y, proba) -> dict[str, float]:
    """Seuil qui maximise le F1 : à n'appeler QUE sur la validation."""
    precision, rappel, seuils = precision_recall_curve(y, proba)
    f1 = 2 * precision * rappel / np.clip(precision + rappel, 1e-12, None)
    i = int(np.argmax(f1[:-1]))
    return {"seuil": float(seuils[i]), "precision": float(precision[i]),
            "rappel": float(rappel[i]), "f1": float(f1[i])}


def metriques_au_seuil(y, proba, seuil: float) -> dict[str, float]:
    """Ce qui se passe concrètement si on alerte au-dessus du seuil."""
    y, pred = np.asarray(y), (np.asarray(proba) >= seuil).astype(int)
    vp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"precision": vp / max(vp + fp, 1), "rappel": vp / max(vp + fn, 1),
            "alertes": int(pred.sum()), "retards_detectes": vp, "retards_rates": fn}