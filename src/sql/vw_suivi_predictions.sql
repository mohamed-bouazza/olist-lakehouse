-- Vue de restitution pour Power BI : une ligne par commande scorée,
-- avec la prédiction, la vraie réponse et les axes d'analyse.
-- À exécuter une fois dans le SQL Editor de Databricks. La vue se recalcule
-- à chaque lecture : elle reflète toujours le dernier run du job olist-daily.

CREATE OR REPLACE VIEW olist_dev.lakehouse.vw_suivi_predictions AS
SELECT
  p.order_id,
  CAST(date_trunc('week', p.order_approved_at) AS DATE) AS semaine,        -- lundi de la semaine
  p.periode,                                                               -- entrainement / validation / test
  p.proba_retard,
  FLOOR(p.proba_retard * 20) / 20                      AS tranche_proba,   -- tranches de 0,05
  CAST(p.alerte AS INT)                                AS alerte,          -- 1 si proba >= seuil
  p.seuil,
  p.version_modele,
  CAST(f.en_retard AS INT)                             AS en_retard,       -- vraie réponse, vide si non livrée
  f.dans_population,                                                       -- vrai si la réponse est fiable
  f.etat_client,
  f.etat_vendeur,
  f.categorie_principale,
  f.type_paiement_principal,
  f.distance_max_km
FROM olist_dev.lakehouse.gold_delivery_predictions AS p
JOIN olist_dev.ml.features_commandes AS f
  ON p.order_id = f.order_id;
