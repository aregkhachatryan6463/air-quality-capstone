from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class DistrictClusteringResult:
    mapping: pd.DataFrame
    method: str
    n_districts: int
    diagnostics: dict[str, Any]


YEREVAN_DISTRICT_ORDER = [
    "ajapnyak",
    "arabkir",
    "avan",
    "davtashen",
    "erebuni",
    "kanaker-zeytun",
    "kentron",
    "malatia-sebastia",
    "nork-marash",
    "nor-nork",
    "nubarashen",
    "shengavit",
]


def build_district_mapping(
    station_meta: pd.DataFrame | None,
    *,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    station_id_col: str = "station_id",
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> DistrictClusteringResult:
    # Adaptive fallback when station metadata is unavailable.
    if station_meta is None or station_meta.empty:
        mapping = pd.DataFrame({"station_id": ["city_aggregate"], "district_id": [0]})
        return DistrictClusteringResult(
            mapping=mapping,
            method="fallback_single_group",
            n_districts=1,
            diagnostics={"reason": "station metadata missing"},
        )

    if "district_slug" in station_meta.columns and station_id_col in station_meta.columns:
        out = station_meta[[station_id_col, "district_slug"]].copy()
        out["district_slug"] = out["district_slug"].astype(str).str.strip().str.lower()
        district_to_id = {d: i for i, d in enumerate(YEREVAN_DISTRICT_ORDER)}
        out = out[out["district_slug"].isin(district_to_id.keys())]
        out = out.drop_duplicates(subset=[station_id_col], keep="first")
        if not out.empty:
            out["district_id"] = out["district_slug"].map(district_to_id).astype(int)
            out = out.rename(columns={"district_slug": "district_name"})
            return DistrictClusteringResult(
                mapping=out[[station_id_col, "district_id", "district_name"]].reset_index(drop=True),
                method="yerevan_admin_district",
                n_districts=len(YEREVAN_DISTRICT_ORDER),
                diagnostics={
                    "district_order": YEREVAN_DISTRICT_ORDER,
                    "n_stations_mapped": int(len(out)),
                    "n_districts_present": int(out["district_id"].nunique()),
                },
            )

    req = {lon_col, lat_col, station_id_col}
    if not req.issubset(station_meta.columns):
        raise ValueError(f"station metadata must include {req}")
    coords = station_meta[[lon_col, lat_col]].astype(float).to_numpy()
    best = None
    rows = []
    k_max = min(max_k, max(min_k, len(station_meta) - 1))
    for k in range(min_k, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(coords)
        score = silhouette_score(coords, labels) if len(np.unique(labels)) > 1 else -1.0
        rows.append({"k": k, "silhouette": float(score)})
        if best is None or score > best["score"]:
            best = {"k": k, "labels": labels, "score": score}
    if best is None:
        mapping = pd.DataFrame({station_id_col: station_meta[station_id_col], "district_id": 0})
        return DistrictClusteringResult(mapping=mapping, method="fallback_single_group", n_districts=1, diagnostics={"reason": "no valid clustering"})

    mapping = pd.DataFrame(
        {
            station_id_col: station_meta[station_id_col].values,
            "district_id": best["labels"].astype(int),
        }
    )
    return DistrictClusteringResult(
        mapping=mapping,
        method="kmeans_geo",
        n_districts=int(best["k"]),
        diagnostics={"silhouette_scan": rows, "best_score": float(best["score"])},
    )

