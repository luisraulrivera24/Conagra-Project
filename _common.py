"""
_common.py -- shared engine for the report template.

Files beginning with "_" are ignored by Quarto as inputs, so this module is a
safe place to keep code shared across reports.

NOTHING HERE TOUCHES THE NETWORK OR THE DISK.
Every dataset is generated on the fly from a fixed seed, so each render is
reproducible and the project has no external data dependencies.

Hard requirements : pandas, numpy, scikit-learn
Optional (auto-fallback if absent) : xgboost, folium, matplotlib

To repoint this template at real data, replace make_records() and make_zones()
with your own loaders and keep the returned column names the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Optional dependency probes. Import failures are recorded, never raised, so a
# missing package degrades one figure instead of killing the whole render.
# --------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import folium
    from folium.features import GeoJson, GeoJsonTooltip
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


SEED = 42
N_RECORDS = 3000
GRID_COLS, GRID_ROWS = 10, 6          # 60 synthetic zones
LON_MIN, LON_MAX = -104.0, -94.0      # arbitrary bounding box
LAT_MIN, LAT_MAX = 26.0, 36.0

CATEGORIES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


# ==========================================================================
# 1. Synthetic geography
# ==========================================================================
def make_zones() -> tuple[dict, pd.DataFrame]:
    """Build a grid of rectangular zones.

    Returns
    -------
    geojson : dict
        A GeoJSON FeatureCollection. Plain Python dicts -- no geopandas or
        shapely needed. Folium accepts this shape directly.
    centroids : pandas.DataFrame
        One row per zone: zone_code, zone_centroid_lon, zone_centroid_lat.
    """
    lon_edges = np.linspace(LON_MIN, LON_MAX, GRID_COLS + 1)
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, GRID_ROWS + 1)

    features, rows = [], []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            lon0, lon1 = lon_edges[col], lon_edges[col + 1]
            lat0, lat1 = lat_edges[row], lat_edges[row + 1]
            zone_code = f"Z{1000 + row * GRID_COLS + col}"

            # GeoJSON rings are [lon, lat] and must close back on themselves.
            ring = [
                [lon0, lat0], [lon1, lat0],
                [lon1, lat1], [lon0, lat1],
                [lon0, lat0],
            ]
            features.append({
                "type": "Feature",
                "properties": {"zone_code": zone_code},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
            rows.append({
                "zone_code": zone_code,
                "zone_centroid_lon": (lon0 + lon1) / 2,
                "zone_centroid_lat": (lat0 + lat1) / 2,
            })

    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, pd.DataFrame(rows)


# ==========================================================================
# 2. Synthetic records
# ==========================================================================
def make_records(n: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """Generate a tabular dataset with a learnable signal plus noise.

    The target is built from the features on purpose: it gives the model
    something real to find, so the metrics in the report look plausible
    instead of random.
    """
    rng = np.random.default_rng(seed)
    _, centroids = make_zones()

    zone_idx = rng.integers(0, len(centroids), size=n)
    zone_code = centroids["zone_code"].to_numpy()[zone_idx]

    # A stable per-zone offset so location genuinely carries signal.
    zone_effect = rng.normal(0, 0.45, size=len(centroids))[zone_idx]

    metric_a = rng.lognormal(mean=7.4, sigma=0.42, size=n).round(0)   # e.g. area
    metric_b = (metric_a * rng.uniform(1.5, 9.0, size=n)).round(1)    # e.g. lot
    count_a = rng.integers(1, 7, size=n)                              # e.g. beds
    count_b = np.clip(count_a - rng.integers(0, 3, size=n), 1, None)  # e.g. baths
    category = rng.choice(CATEGORIES, size=n, p=[.34, .22, .16, .12, .09, .07])

    category_effect = pd.Series(category).map(
        dict(zip(CATEGORIES, [0.00, -0.18, 0.12, -0.35, 0.28, -0.08]))
    ).to_numpy()

    log_value = (
        6.9
        + 0.55 * np.log(metric_a)
        + 0.08 * np.log(metric_b)
        + 0.05 * count_a
        + 0.04 * count_b
        + zone_effect
        + category_effect
        + rng.normal(0, 0.16, size=n)   # tuned so R^2 lands near 0.86
    )

    df = pd.DataFrame({
        "zone_code": zone_code,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "count_a": count_a,
        "count_b": count_b,
        "category": category,
        "value": np.expm1(log_value).round(2),
    })

    # Sprinkle missing values so the template exercises NA-tolerant paths.
    df.loc[rng.choice(n, size=int(n * 0.03), replace=False), "metric_b"] = np.nan
    return df


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Merge on geography and one-hot encode. Mirrors a real cleaning step."""
    _, centroids = make_zones()
    out = df.merge(centroids, on="zone_code", how="left")
    out = pd.get_dummies(out, columns=["category"], dtype="int")
    return out


# ==========================================================================
# 3. Model
# ==========================================================================
def fit_model(frame: pd.DataFrame, seed: int = SEED):
    """Train a gradient-boosted regressor on log1p(value).

    Uses XGBoost when installed and falls back to scikit-learn's
    HistGradientBoostingRegressor otherwise. Both tolerate NaN features.

    Returns a dict of metrics, predictions, and the engine name used.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
    )

    X = frame.drop(columns=["value"])
    y = np.log1p(frame["value"])

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    # Keep zone codes aside for aggregation, out of the feature matrix.
    zone_test = X_test_full["zone_code"].reset_index(drop=True)
    X_train = X_train_full.drop(columns=["zone_code"])
    X_test = X_test_full.drop(columns=["zone_code"])

    if HAS_XGBOOST:
        engine = "XGBoost"
        model = XGBRegressor(
            n_estimators=350, max_depth=6, learning_rate=0.05,
            subsample=0.9, random_state=seed,
        )
    else:
        engine = "scikit-learn HistGradientBoosting"
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=350, max_depth=6, learning_rate=0.05, random_state=seed,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return {
        "engine": engine,
        "model": model,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "zone_test": zone_test,
        "metrics": {
            "MSE": round(float(mse), 2),
            "RMSE": round(float(np.sqrt(mse)), 2),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "R2": round(float(r2_score(y_test, y_pred)), 2),
        },
    }


def aggregate_by_zone(fit: dict) -> pd.DataFrame:
    """Average predicted vs actual per zone, on both log and natural scales."""
    res = pd.DataFrame({
        "zone_code": fit["zone_test"],
        "prediction": fit["y_pred"],
        "actual": fit["y_test"].reset_index(drop=True),
    })
    res = res.groupby("zone_code")[["prediction", "actual"]].mean().reset_index()
    res = res.rename(columns={
        "prediction": "avg_prediction_log",
        "actual": "avg_actual_log",
    })
    res["avg_prediction"] = np.expm1(res["avg_prediction_log"]).round(2)
    res["avg_actual"] = np.expm1(res["avg_actual_log"]).round(2)
    res["avg_prediction_log"] = res["avg_prediction_log"].round(3)
    res["avg_actual_log"] = res["avg_actual_log"].round(3)
    return res


# ==========================================================================
# 4. Map rendering
# ==========================================================================
def _geojson_with_values(geojson: dict, results: pd.DataFrame) -> dict:
    """Copy the GeoJSON and inject per-zone values into feature properties.

    Writing values straight into properties means the tooltip reads from the
    same object the choropleth colours -- no join to fall out of sync.
    """
    lookup = results.set_index("zone_code").to_dict(orient="index")
    out = {"type": "FeatureCollection", "features": []}
    for feat in geojson["features"]:
        code = feat["properties"]["zone_code"]
        if code not in lookup:
            continue                      # zone absent from the test split
        props = {"zone_code": code}
        props.update({k: v for k, v in lookup[code].items()})
        out["features"].append({
            "type": "Feature",
            "properties": props,
            "geometry": feat["geometry"],
        })
    return out


def _folium_panel(gj: dict, results: pd.DataFrame, value_col: str,
                  label_col: str, legend: str, alias: str):
    """One choropleth panel."""
    m = folium.Map(location=[(LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2],
                   zoom_start=5, tiles="openstreetmap")

    folium.Choropleth(
        geo_data=gj,
        data=results,
        columns=["zone_code", value_col],
        key_on="feature.properties.zone_code",
        fill_color="plasma",
        fill_opacity=0.7,
        line_opacity=0.45,
        legend_name=legend,
    ).add_to(m)

    GeoJson(
        gj,
        style_function=lambda _: {
            "fillColor": "transparent", "color": "transparent", "weight": 0
        },
        tooltip=GeoJsonTooltip(
            fields=["zone_code", label_col],
            aliases=["Zone:", alias],
            localize=True, sticky=True, labels=True,
        ),
    ).add_to(m)
    return m


def _matplotlib_fallback(geojson: dict, results: pd.DataFrame):
    """Static two-panel choropleth for environments without folium."""
    lookup = results.set_index("zone_code")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    panels = [
        ("avg_prediction_log", "Predicted (log scale)"),
        ("avg_actual_log", "Actual (log scale)"),
    ]
    vmin = float(results[["avg_prediction_log", "avg_actual_log"]].min().min())
    vmax = float(results[["avg_prediction_log", "avg_actual_log"]].max().max())
    cmap = plt.get_cmap("plasma")

    for ax, (col, title) in zip(axes, panels):
        for feat in geojson["features"]:
            code = feat["properties"]["zone_code"]
            ring = feat["geometry"]["coordinates"][0]
            if code in lookup.index:
                frac = (lookup.loc[code, col] - vmin) / (vmax - vmin or 1)
                face = cmap(float(frac))
            else:
                face = "#eeeeee"
            ax.add_patch(MplPolygon(ring, closed=True, facecolor=face,
                                    edgecolor="white", linewidth=0.6))
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)

    fig.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes, fraction=0.025, pad=0.02, label="Log scale",
    )
    return fig


def dual_map(results: pd.DataFrame, geojson: dict):
    """Side-by-side predicted vs actual choropleths.

    Returns an IPython HTML object when folium is available, a matplotlib
    figure when it is not, and a short warning string if neither is installed.
    Quarto renders all three correctly.
    """
    if HAS_FOLIUM:
        gj = _geojson_with_values(geojson, results)
        m1 = _folium_panel(gj, results, "avg_prediction_log", "avg_prediction",
                           "Predicted (log scale)", "Predicted:")
        m2 = _folium_panel(gj, results, "avg_actual_log", "avg_actual",
                           "Actual (log scale)", "Actual:")
        html = f"""
        <div style="display:flex; gap:1rem; flex-wrap:wrap;">
          <div style="flex:1 1 320px;">
            <h4 style="margin:.2rem 0;">Predicted</h4>{m1._repr_html_()}
          </div>
          <div style="flex:1 1 320px;">
            <h4 style="margin:.2rem 0;">Actual</h4>{m2._repr_html_()}
          </div>
        </div>"""
        from IPython.display import HTML
        return HTML(html)

    if HAS_MATPLOTLIB:
        _, _ = None, None
        return _matplotlib_fallback(geojson, results)

    return "Install `folium` or `matplotlib` to render the map."


# ==========================================================================
# 5. One-call convenience wrapper
# ==========================================================================
def build_everything(seed: int = SEED) -> dict:
    """Run the whole pipeline. Handy for reports that only need the outputs."""
    geojson, centroids = make_zones()
    raw = make_records(seed=seed)
    frame = prepare_frame(raw)
    fit = fit_model(frame, seed=seed)
    results = aggregate_by_zone(fit)
    return {
        "geojson": geojson, "centroids": centroids, "raw": raw,
        "frame": frame, "fit": fit, "results": results,
    }
