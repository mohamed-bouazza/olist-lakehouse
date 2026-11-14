"""Calculs géographiques."""

from pyspark.sql import Column
from pyspark.sql import functions as F

RAYON_TERRE_KM = 6371.0


def distance_km(lat1, lng1, lat2, lng2) -> Column:
    """Distance à vol d'oiseau entre deux points (formule de haversine), en km.

    Accepte des noms de colonnes ou des colonnes. Renvoie null si une coordonnée manque.
    """
    lat1, lng1, lat2, lng2 = [F.col(c) if isinstance(c, str) else c for c in (lat1, lng1, lat2, lng2)]
    dlat, dlng = F.radians(lat2 - lat1), F.radians(lng2 - lng1)
    a = F.sin(dlat / 2) ** 2 + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlng / 2) ** 2
    return 2 * RAYON_TERRE_KM * F.asin(F.sqrt(a))