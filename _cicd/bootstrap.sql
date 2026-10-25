-- Amorçage unique de la plateforme : objets durables, hors bundle.
CREATE CATALOG IF NOT EXISTS olist_dev;
CREATE CATALOG IF NOT EXISTS olist_prod;

-- Système source simulé (lu uniquement par le rejoueur)
CREATE SCHEMA IF NOT EXISTS olist_dev.raw;
CREATE SCHEMA IF NOT EXISTS olist_prod.raw;
CREATE VOLUME IF NOT EXISTS olist_dev.raw.source_olist;
CREATE VOLUME IF NOT EXISTS olist_prod.raw.source_olist;