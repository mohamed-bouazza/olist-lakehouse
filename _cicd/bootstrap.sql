-- Amorçage de la plateforme : objets durables, hors bundle.
CREATE CATALOG IF NOT EXISTS olist_dev;
CREATE CATALOG IF NOT EXISTS olist_prod;

CREATE SCHEMA IF NOT EXISTS olist_dev.raw;        -- système source simulé
CREATE SCHEMA IF NOT EXISTS olist_dev.lakehouse;  -- bronze, silver, gold
CREATE SCHEMA IF NOT EXISTS olist_dev.ops;        -- état du rejoueur, métriques
CREATE SCHEMA IF NOT EXISTS olist_dev.ml;         -- features, prédictions, modèles
CREATE VOLUME IF NOT EXISTS olist_dev.raw.source_olist;
CREATE VOLUME IF NOT EXISTS olist_dev.raw.landing;

CREATE SCHEMA IF NOT EXISTS olist_prod.raw;
CREATE SCHEMA IF NOT EXISTS olist_prod.lakehouse;
CREATE SCHEMA IF NOT EXISTS olist_prod.ops;
CREATE SCHEMA IF NOT EXISTS olist_prod.ml;
CREATE VOLUME IF NOT EXISTS olist_prod.raw.source_olist;
CREATE VOLUME IF NOT EXISTS olist_prod.raw.landing;