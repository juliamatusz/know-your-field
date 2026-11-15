"""Know Your Field – shared logic for the Polish (app_rolnik.py) and English (app_farmer_en.py) apps.

Reads only files bundled in this folder:
  data/dataset/dane_dzienne.csv     daily weather, soil moisture and Sentinel-2 indices per field
  data/dataset/pola_statyczne.csv   static field attributes (area, elevation, slope)
  data/dataset/s2_obserwacje.csv    Sentinel-2 observations per field
  data/model/predykcje.csv          crop water-stress predictions (optional)
  pola/*.geojson                    field boundaries
Column names in the CSV files are in Polish; user-facing text is translated in T below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WLKP_URL = "https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-min.geojson"

# soil-moisture series, best first: ERA5-Land (m3/m3, ~9 km), MERRA-2 (index 0–1, ~50 km)
SOIL_SOURCES = [("wilg_gleby_0_7cm_m3m3", "ERA5-Land"), ("wilg_gleby_wierzch_0_1", "MERRA-2")]
WINDOW = 15  # ± days around the same date in other years = "normal for this time of year"

COLORS = {"red": "#c0392b", "yellow": "#e5a50a", "green": "#2e8b57", "blue": "#2c7fb8", "grey": "#8c8c8c"}

T = {
    "pl": {
        "title": "Know Your Field",
        "subtitle": "Wielkopolska · podpowiedź z danych satelitarnych i pogodowych – nie zastępuje oglądania pola",
        "day": "Stan na dzień",
        "field": "Pole",
        "pick": "Wybierz pole (lub kliknij na mapie)",
        "levels": {"red": "Wyraźnie sucho – warto sprawdzić pole", "yellow": "Lekko sucho – obserwuj",
                   "green": "Woda w normie", "blue": "Bardzo mokro", "grey": "Brak danych na ten dzień"},
        "legend": {"green": "w normie", "yellow": "lekko sucho", "red": "wyraźnie sucho", "blue": "bardzo mokro"},
        "pct_very_dry": "Gleba bardzo sucha jak na tę porę roku ({p:.0f}. percentyl).",
        "pct_dry": "Gleba suchsza niż zwykle ({p:.0f}. percentyl).",
        "pct_wet": "Gleba bardzo mokra ({p:.0f}. percentyl) – możliwe zastoiska wody.",
        "pct_ok": "Wilgotność gleby typowa dla tej pory roku ({p:.0f}. percentyl).",
        "bil_deficit": "W ostatnich 30 dniach brakło ok. {v:.0f} mm wody (zużycie > opady).",
        "bil_surplus": "W ostatnich 30 dniach opady przewyższyły zużycie o ok. {v:.0f} mm.",
        "pred_stress": "Zdjęcie z {d:%d.%m}: woda w liściach niższa niż oczekiwana – możliwy stres.",
        "pred_ok": "Zdjęcie z {d:%d.%m}: stan roślin zgodny z oczekiwaniem.",
        "frozen": "Gleba prawdopodobnie zamarznięta – odczyty wilgotności są wtedy niepewne.",
        "m_pct": "Wilgotność vs norma", "m_pct_help": "0 = najsucho w tej porze roku, 50 = typowo, 100 = najmokrzej",
        "m_bil": "Bilans wody 30 dni", "m_bil_help": "opady minus zużycie wody (ujemny = niedobór)",
        "m_s2": "Ostatnie zdjęcie satelitarne",
        "tabs": ["💧 Wilgotność gleby", "🌱 Rośliny", "🌦 Opady", "🔎 Nietypowe miesiące"],
        "an_title": "Jak nietypowy był każdy miesiąc na tym polu (2020–2025)",
        "an_rank": "nietypowość (0–100)", "an_flag": "miesiąc nietypowy",
        "an_m_count": "Nietypowe miesiące", "an_m_count_help": "liczba miesięcy oznaczonych przez model na tym polu",
        "an_m_now": "Nietypowość miesiąca {m}", "an_m_now_help": "0 = najbardziej typowy miesiąc, 100 = najbardziej nietypowy ze wszystkich pól i lat",
        "an_top": "Najbardziej nietypowe miesiące na tym polu",
        "an_drivers_none": "brak jednej wyraźnej przyczyny – nietypowe połączenie warunków",
        "an_note": "Model (Isolation Forest) porównuje pogodę, wilgotność gleby i stan roślin ze wszystkich pól "
                   "i lat. ↑ / ↓ = wyżej / niżej niż w tym samym miesiącu innych lat. "
                   "Nietypowy miesiąc to sygnał do sprawdzenia, a nie diagnoza suszy czy szkody.",
        "an_none": "Za mało danych, żeby wykryć nietypowe miesiące.",
        "features": {
            "t_srednia_c": "temperatura średnia", "t_max_c": "temperatura maks.", "t_min_c": "temperatura min.",
            "opad_mm": "opady", "et0_mm": "parowanie (ET0)", "bilans_uprawy_30d_mm": "bilans wody 30 dni",
            "wilg_gleby_wierzch_0_1": "wilgotność gleby (wierzch)", "wilg_gleby_korzen_0_1": "wilgotność gleby (korzenie)",
            "smap_m3m3": "wilgotność gleby (SMAP)", "vpd_kpa": "suchość powietrza (VPD)",
            "NDVI_interp": "zieleń (NDVI)", "NDMI_interp": "woda w liściach (NDMI)", "NDRE_interp": "chlorofil (NDRE)",
            "EVI_interp": "biomasa (EVI)", "BSI_interp": "odsłonięta gleba (BSI)",
            "wilg_gleby_wierzch_0_1_zmiana_7d": "zmiana wilgotności wierzchu (7 dni)",
            "wilg_gleby_korzen_0_1_zmiana_7d": "zmiana wilgotności przy korzeniach (7 dni)",
        },
        "soil_title": "Wilgotność gleby w {y} na tle lat 2020–2025",
        "soil_band": "typowy zakres", "soil_line": "wilgotność ({src})",
        "soil_note": "Szare pasmo = typowy zakres dla tej pory roku. Linia poniżej pasma = susza.",
        "plant_title": "Stan roślin z Sentinel-2 – {y}",
        "ndvi": "zieleń (NDVI)", "ndmi": "woda w liściach (NDMI)", "stress": "możliwy stres wodny",
        "plant_note": "Spadek obu linii w sezonie = rośliny słabną lub są zbierane.",
        "rain_title": "Opady w {y} vs średnia innych lat",
        "rain_cur": "opady {y}", "rain_avg": "średnia", "mm": "mm / miesiąc",
        "months": ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"],
        "no_data": "Brak danych.", "missing": "Brak plików z danymi w folderze data/.",
        "footer": "PoC · dane: NASA POWER / MERRA-2, ERA5-Land, ESA Sentinel-2, SRTM. "
                  "Narzędzie wspiera, a nie zastępuje decyzje rolnika.",
    },
    "en": {
        "title": "Know Your Field",
        "subtitle": "Wielkopolska · guidance from satellite and weather data – it does not replace looking at the field",
        "day": "Status on",
        "field": "Field",
        "pick": "Pick a field (or click on the map)",
        "levels": {"red": "Clearly dry – worth checking the field", "yellow": "Slightly dry – keep an eye on it",
                   "green": "Water is normal", "blue": "Very wet", "grey": "No data for this day"},
        "legend": {"green": "normal", "yellow": "slightly dry", "red": "clearly dry", "blue": "very wet"},
        "pct_very_dry": "Soil is very dry for this time of year ({p:.0f}th percentile).",
        "pct_dry": "Soil is drier than usual ({p:.0f}th percentile).",
        "pct_wet": "Soil is very wet ({p:.0f}th percentile) – waterlogging is possible.",
        "pct_ok": "Soil moisture is normal for this time of year ({p:.0f}th percentile).",
        "bil_deficit": "Over the last 30 days the field lacked about {v:.0f} mm of water (use > rainfall).",
        "bil_surplus": "Over the last 30 days rainfall exceeded water use by about {v:.0f} mm.",
        "pred_stress": "Image from {d:%d %b}: leaf water lower than expected – possible stress.",
        "pred_ok": "Image from {d:%d %b}: crop condition as expected.",
        "frozen": "Soil is probably frozen – moisture readings are unreliable then.",
        "m_pct": "Moisture vs normal", "m_pct_help": "0 = driest for this time of year, 50 = typical, 100 = wettest",
        "m_bil": "30-day water balance", "m_bil_help": "rainfall minus water use (negative = deficit)",
        "m_s2": "Latest satellite image",
        "tabs": ["💧 Soil water", "🌱 Crops", "🌦 Rainfall", "🔎 Unusual months"],
        "an_title": "How unusual each month was on this field (2020–2025)",
        "an_rank": "unusualness (0–100)", "an_flag": "unusual month",
        "an_m_count": "Unusual months", "an_m_count_help": "number of months flagged by the model on this field",
        "an_m_now": "Unusualness of {m}", "an_m_now_help": "0 = most typical month, 100 = most unusual across all fields and years",
        "an_top": "Most unusual months on this field",
        "an_drivers_none": "no single clear cause – an unusual combination of conditions",
        "an_note": "The model (Isolation Forest) compares weather, soil moisture and crop condition across all fields "
                   "and years. ↑ / ↓ = higher / lower than in the same month of other years. "
                   "An unusual month is a prompt to check, not a diagnosis of drought or damage.",
        "an_none": "Not enough data to detect unusual months.",
        "features": {
            "t_srednia_c": "mean temperature", "t_max_c": "max temperature", "t_min_c": "min temperature",
            "opad_mm": "rainfall", "et0_mm": "evaporation (ET0)", "bilans_uprawy_30d_mm": "30-day water balance",
            "wilg_gleby_wierzch_0_1": "soil moisture (top)", "wilg_gleby_korzen_0_1": "soil moisture (roots)",
            "smap_m3m3": "soil moisture (SMAP)", "vpd_kpa": "air dryness (VPD)",
            "NDVI_interp": "green cover (NDVI)", "NDMI_interp": "leaf water (NDMI)", "NDRE_interp": "chlorophyll (NDRE)",
            "EVI_interp": "biomass (EVI)", "BSI_interp": "bare soil (BSI)",
            "wilg_gleby_wierzch_0_1_zmiana_7d": "top-soil moisture change (7 days)",
            "wilg_gleby_korzen_0_1_zmiana_7d": "root-zone moisture change (7 days)",
        },
        "soil_title": "Soil moisture in {y} against 2020–2025",
        "soil_band": "typical range", "soil_line": "moisture ({src})",
        "soil_note": "Grey band = typical range for this time of year. Line below the band = drought.",
        "plant_title": "Crop condition from Sentinel-2 – {y}",
        "ndvi": "green cover (NDVI)", "ndmi": "leaf water (NDMI)", "stress": "possible water stress",
        "plant_note": "Both lines dropping in season = crops weakening or being harvested.",
        "rain_title": "Rainfall in {y} vs other years",
        "rain_cur": "rainfall {y}", "rain_avg": "average", "mm": "mm / month",
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "no_data": "No data.", "missing": "Data files not found in the data/ folder.",
        "footer": "PoC · data: NASA POWER / MERRA-2, ERA5-Land, ESA Sentinel-2, SRTM. "
                  "This tool supports, and does not replace, the farmer's decisions.",
    },
}


# =================================================================== data

def _doy_dist(doy: np.ndarray, d: int) -> np.ndarray:
    diff = np.abs(doy - d)
    return np.minimum(diff, 366 - diff)


def soil_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Per field and day: soil moisture, normal range (p10/p50/p90) and percentile for the time of year."""
    col, src = next(((c, s) for c, s in SOIL_SOURCES if c in daily and daily[c].notna().sum() > 365), (None, None))
    if col is None:
        return pd.DataFrame()
    parts = []
    for field, d in daily.groupby("field"):
        d = d.sort_values("date")
        t = pd.DataFrame({"field": field, "date": d["date"].values, "value": d[col].values, "source": src})
        doy = t["date"].dt.dayofyear.values
        ok = t["value"].notna().values
        doy_ok, val_ok = doy[ok], t["value"].values[ok]
        q = np.full((367, 3), np.nan)
        pct = np.full(len(t), np.nan)
        for day in range(1, 367):
            ref = np.sort(val_ok[_doy_dist(doy_ok, day) <= WINDOW])
            if ref.size < 20:
                continue
            q[day] = np.quantile(ref, [0.10, 0.50, 0.90])
            idx = np.where(ok & (doy == day))[0]
            pct[idx] = 100 * np.searchsorted(ref, t["value"].values[idx], side="right") / ref.size
        t["p10"], t["p50"], t["p90"] = q[doy, 0], q[doy, 1], q[doy, 2]
        t["pct"] = pct
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_all():
    ds = DATA / "dataset"
    daily = pd.read_csv(ds / "dane_dzienne.csv", parse_dates=["date"])
    static = pd.read_csv(ds / "pola_statyczne.csv")
    s2 = pd.read_csv(ds / "s2_obserwacje.csv", parse_dates=["date"])
    pred_path = DATA / "model" / "predykcje.csv"
    pred = pd.read_csv(pred_path, parse_dates=["date"]) if pred_path.exists() else None
    geoms = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "pola").glob("*.geojson"))}
    return daily, static, s2, pred, geoms, soil_table(daily)


# ============================================================ AI anomaly detection

ML_FEATURES = [
    "t_srednia_c",
    "t_max_c",
    "t_min_c",
    "opad_mm",
    "et0_mm",
    "bilans_uprawy_30d_mm",
    "wilg_gleby_wierzch_0_1",
    "wilg_gleby_korzen_0_1",
    "smap_m3m3",
    "vpd_kpa",
    "NDVI_interp",
    "NDMI_interp",
    "NDRE_interp",
    "EVI_interp",
    "BSI_interp",
    "wilg_gleby_wierzch_0_1_zmiana_7d",
    "wilg_gleby_korzen_0_1_zmiana_7d",
]

# Sum rainfall; average the other variables.
SUM_FEATURES = {"opad_mm"}

N_DRIVERS = 3      # how many main deviations to show per month
DRIVER_MIN_Z = 1.5  # |z| below this is not reported as a cause


@st.cache_data(show_spinner=False)
def detect_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly environmental anomaly detection.

    One row = one field and one calendar month.
    Isolation Forest is trained jointly across all fields.
    Column "drivers": list of (feature, z) – the largest deviations from the same
    calendar month in other years on the same field (explanation only, not used by the model).
    """

    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Use the available water-balance column (fallback renamed so the rest of the code sees one name).
    if "bilans_uprawy_30d_mm" not in df.columns and "bilans_30d_mm" in df.columns:
        df = df.rename(columns={"bilans_30d_mm": "bilans_uprawy_30d_mm"})

    available = [c for c in ML_FEATURES if c in df.columns]
    if len(available) < 3:
        return pd.DataFrame()

    # Monthly aggregation.
    agg = {col: "sum" if col in SUM_FEATURES else "mean" for col in available}
    monthly = df.groupby(["field", "year", "month"]).agg(agg).reset_index()

    # Keep only usable numerical features.
    features = [
        c for c in available
        if monthly[c].notna().sum() >= 30 and monthly[c].nunique(dropna=True) > 1
    ]
    if len(monthly) < 30 or len(features) < 3:
        return pd.DataFrame()

    # Encode seasonality cyclically.
    # This helps distinguish normal seasonal changes from unusual environmental conditions.
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)

    X = monthly[features + ["month_sin", "month_cos"]].replace([np.inf, -np.inf], np.nan)
    # Avoid columns that contain only missing values.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] < 3:
        return pd.DataFrame()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=200, contamination=0.08, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X)

    monthly["anomaly"] = pipeline.predict(X) == -1
    monthly["anomaly_score"] = -pipeline.score_samples(X)
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
    )
    # Relative ranking, not probability.
    monthly["anomaly_rank"] = monthly["anomaly_score"].rank(pct=True) * 100

    # Explanation: deviation from the same calendar month in the other years on the same field.
    grp = monthly.groupby(["field", "month"])[features]
    z = (monthly[features] - grp.transform("mean")) / grp.transform("std")
    z = z.replace([np.inf, -np.inf], np.nan)

    def top_drivers(row: pd.Series) -> list[tuple[str, float]]:
        r = row.dropna()
        r = r[r.abs() >= DRIVER_MIN_Z]
        r = r.reindex(r.abs().sort_values(ascending=False).index)
        return [(k, float(v)) for k, v in r.head(N_DRIVERS).items()]

    monthly["drivers"] = z.apply(top_drivers, axis=1)
    return monthly


@st.cache_data(show_spinner=False)
def load_wielkopolska():
    local = DATA / "wielkopolska.geojson"
    try:
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))
        import requests

        gj = requests.get(WLKP_URL, timeout=10).json()
        feats = [f for f in gj["features"]
                 if "wielkopol" in json.dumps(f.get("properties", {}), ensure_ascii=False).lower()]
        return {"type": "FeatureCollection", "features": feats} if feats else None
    except Exception:  # noqa: BLE001
        return None


# ============================================================ status logic

def field_status(tx: dict, field: str, day: pd.Timestamp, daily, soil, pred) -> dict:
    d = daily[(daily["field"] == field) & (daily["date"] == day)]
    s = soil[(soil["field"] == field) & (soil["date"] == day)] if not soil.empty else soil
    reasons, score = [], 0

    pct = s["pct"].iloc[0] if len(s) else np.nan
    if pd.notna(pct):
        if pct < 10:
            score += 2; reasons.append(tx["pct_very_dry"].format(p=pct))
        elif pct < 25:
            score += 1; reasons.append(tx["pct_dry"].format(p=pct))
        elif pct > 90:
            reasons.append(tx["pct_wet"].format(p=pct))
        else:
            reasons.append(tx["pct_ok"].format(p=pct))

    bil_col = "bilans_uprawy_30d_mm" if "bilans_uprawy_30d_mm" in daily else "bilans_30d_mm"
    bil = d[bil_col].iloc[0] if len(d) and bil_col in d else np.nan
    if pd.notna(bil):
        if bil < -80:
            score += 2; reasons.append(tx["bil_deficit"].format(v=-bil))
        elif bil < -40:
            score += 1; reasons.append(tx["bil_deficit"].format(v=-bil))
        elif bil > 20:
            reasons.append(tx["bil_surplus"].format(v=bil))

    if pred is not None:
        p = pred[(pred["field"] == field) & pred["date"].between(day - pd.Timedelta(days=20), day)]
        if len(p):
            last = p.sort_values("date").iloc[-1]
            if last["ocena"] == "prawdopodobny_stres_wodny":
                score += 1; reasons.append(tx["pred_stress"].format(d=last["date"]))
            elif not str(last["ocena"]).startswith("gwaltowny"):
                reasons.append(tx["pred_ok"].format(d=last["date"]))

    if len(d) and "t_gleby_0_7cm_c" in d and d["t_gleby_0_7cm_c"].iloc[0] <= 0.5:
        reasons.append(tx["frozen"])

    if not reasons:
        level = "grey"
    elif pd.notna(pct) and pct > 90 and score == 0:
        level = "blue"
    elif score >= 3:
        level = "red"
    elif score >= 1:
        level = "yellow"
    else:
        level = "green"
    return {"level": level, "reasons": reasons, "pct": pct, "bil": bil}


# ==================================================================== map

def _center(gj: dict) -> tuple[float, float]:
    c = np.array(gj["features"][0]["geometry"]["coordinates"][0])
    return c[:, 1].mean(), c[:, 0].mean()


def draw_map(geoms: dict, statuses: dict, selected: str, label) -> str | None:
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[52.35, 17.25], zoom_start=8, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics", name="Satellite", max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="&copy; Esri", name="Labels", overlay=True, control=False, max_zoom=19,
    ).add_to(m)
    wlkp = load_wielkopolska()
    if wlkp:
        folium.GeoJson(wlkp, style_function=lambda _: {"color": "#ffd54f", "weight": 2, "fillOpacity": 0}).add_to(m)
    for name, gj in geoms.items():
        color = COLORS[statuses.get(name, {}).get("level", "grey")]
        folium.GeoJson(gj, style_function=lambda _, c=color: {"color": c, "weight": 2, "fillColor": c, "fillOpacity": 0.3},
                       tooltip=label(name)).add_to(m)
        folium.CircleMarker(_center(gj), radius=12 if name == selected else 8, color="white", weight=2,
                            fill=True, fill_color=color, fill_opacity=0.95, tooltip=label(name)).add_to(m)
    out = st_folium(m, height=520, use_container_width=True, returned_objects=["last_object_clicked"])
    click = (out or {}).get("last_object_clicked")
    if click:
        dist = {n: (_center(g)[0] - click["lat"]) ** 2 + (_center(g)[1] - click["lng"]) ** 2 for n, g in geoms.items()}
        best = min(dist, key=dist.get)
        if dist[best] < 0.05:
            return best
    return None


# ================================================================= charts

LAYOUT = dict(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15),
              hovermode="x unified")


def chart_soil(tx, soil_f, year, day):
    t = soil_f[soil_f["date"].dt.year == year]
    if t.empty or t["value"].isna().all():
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["date"], y=t["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["date"], y=t["p10"], fill="tonexty", fillcolor="rgba(160,174,184,0.35)",
                             line=dict(width=0), name=tx["soil_band"], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["date"], y=t["value"], line=dict(color="#1f4e79", width=2),
                             name=tx["soil_line"].format(src=t["source"].iloc[0])))
    fig.add_vline(x=day, line=dict(color="#555", dash="dash"))
    fig.update_layout(title=tx["soil_title"].format(y=year), **LAYOUT)
    return fig


def chart_plants(tx, s2f, predf, year, day):
    obs = s2f[(s2f["date"].dt.year == year) & s2f["qc_ok"]]
    if obs.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs["date"], y=obs["NDVI"], mode="lines+markers", name=tx["ndvi"],
                             line=dict(color="#2e8b57")))
    fig.add_trace(go.Scatter(x=obs["date"], y=obs["NDMI"], mode="lines+markers", name=tx["ndmi"],
                             line=dict(color="#1f77b4")))
    if predf is not None:
        p = predf[(predf["date"].dt.year == year) & (predf["ocena"] == "prawdopodobny_stres_wodny")]
        if len(p):
            fig.add_trace(go.Scatter(x=p["date"], y=p["NDMI"], mode="markers", name=tx["stress"],
                                     marker=dict(size=13, color="#c0392b", symbol="x")))
    fig.add_vline(x=day, line=dict(color="#555", dash="dash"))
    fig.update_layout(title=tx["plant_title"].format(y=year), **LAYOUT)
    return fig


def chart_rain(tx, d_f, year):
    d = d_f.assign(yr=d_f["date"].dt.year, mo=d_f["date"].dt.month)
    m = d.groupby(["yr", "mo"])["opad_mm"].sum().reset_index()
    avg = m[m["yr"] != year].groupby("mo")["opad_mm"].mean().reindex(range(1, 13))
    cur = m[m["yr"] == year].set_index("mo")["opad_mm"].reindex(range(1, 13))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tx["months"], y=cur, name=tx["rain_cur"].format(y=year), marker_color="#2c7fb8"))
    fig.add_trace(go.Scatter(x=tx["months"], y=avg, name=tx["rain_avg"], mode="lines+markers",
                             line=dict(color="#555", dash="dot")))
    fig.update_layout(title=tx["rain_title"].format(y=year), yaxis_title=tx["mm"], **LAYOUT)
    return fig


def chart_anomalies(tx, an_f, day):
    if an_f.empty:
        return None
    an_f = an_f.sort_values("date")
    flagged = an_f[an_f["anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an_f["date"], y=an_f["anomaly_rank"], mode="lines", name=tx["an_rank"],
                             line=dict(color="#8c8c8c", width=1.5)))
    fig.add_trace(go.Scatter(x=flagged["date"], y=flagged["anomaly_rank"], mode="markers", name=tx["an_flag"],
                             marker=dict(size=12, color=COLORS["red"])))
    fig.add_vline(x=day, line=dict(color="#555", dash="dash"))
    fig.update_layout(title=tx["an_title"], yaxis=dict(range=[0, 102]), **LAYOUT)
    return fig


def describe_drivers(tx, drivers) -> str:
    if not drivers:
        return tx["an_drivers_none"]
    return ", ".join(f"{tx['features'].get(k, k)} {'↑' if v > 0 else '↓'}" for k, v in drivers)


# ===================================================================== UI

def run(lang: str) -> None:
    tx = T[lang]
    st.set_page_config(page_title=tx["title"], page_icon="💧", layout="wide", initial_sidebar_state="collapsed")

    def label(name: str) -> str:
        return name.replace("pole", f"{tx['field']} ") if name.startswith("pole") else name

    try:
        daily, static, s2, pred, geoms, soil = load_all()
    except FileNotFoundError:
        st.error(tx["missing"])
        st.stop()
    anomalies = detect_anomalies(daily)

    fields = sorted(daily["field"].unique())
    last_day = daily.dropna(subset=["t_srednia_c"])["date"].max()

    h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
    h1.title(tx["title"])
    h1.caption(tx["subtitle"])
    day = pd.Timestamp(h2.date_input(tx["day"], value=last_day.date(),
                                     min_value=daily["date"].min().date(), max_value=last_day.date()))

    statuses = {f: field_status(tx, f, day, daily, soil, pred) for f in fields}
    st.session_state.setdefault("field", fields[0])

    col_map, col_panel = st.columns([5, 6], gap="large")

    with col_map:
        clicked = draw_map(geoms, statuses, st.session_state.field, label)
        if clicked and clicked != st.session_state.field:
            st.session_state.field = clicked
            st.rerun()
        st.caption("  ".join(f"<span style='color:{COLORS[k]}'>●</span> {v}" for k, v in tx["legend"].items()),
                   unsafe_allow_html=True)

    field = st.session_state.field
    s = statuses[field]
    srow = static[static["field"] == field].iloc[0] if (static["field"] == field).any() else pd.Series(dtype=float)
    d_f = daily[daily["field"] == field]
    soil_f = soil[soil["field"] == field] if not soil.empty else soil
    s2f = s2[s2["field"] == field]
    predf = pred[pred["field"] == field] if pred is not None else None

    with col_panel:
        st.selectbox(tx["pick"], fields, key="field", format_func=label)
        info = [f"{srow['powierzchnia_ha']:.0f} ha" if pd.notna(srow.get("powierzchnia_ha")) else "",
                f"{srow['wys_srednia_m']:.0f} m n.p.m." if pd.notna(srow.get("wys_srednia_m")) else ""]
        if lang == "en":
            info[1] = info[1].replace("n.p.m.", "a.s.l.")

        color = COLORS[s["level"]]
        reasons = "".join(f"<li>{r}</li>" for r in s["reasons"])
        st.markdown(
            f"""<div style="border-left:8px solid {color};background:{color}14;padding:14px 18px;border-radius:6px">
            <div style="font-size:0.85rem;opacity:0.7">{label(field)} · {' · '.join(i for i in info if i)}</div>
            <div style="font-size:1.4rem;font-weight:600;margin-top:2px">{tx['levels'][s['level']]}</div>
            <ul style="margin:8px 0 0 0">{reasons}</ul></div>""",
            unsafe_allow_html=True)
        st.write("")

        m1, m2, m3 = st.columns(3)
        m1.metric(tx["m_pct"], f"{s['pct']:.0f} / 100" if pd.notna(s["pct"]) else "—", help=tx["m_pct_help"])
        m2.metric(tx["m_bil"], f"{s['bil']:.0f} mm" if pd.notna(s["bil"]) else "—", help=tx["m_bil_help"])
        last_obs = s2f[(s2f["date"] <= day) & s2f["qc_ok"]].sort_values("date").tail(1)
        m3.metric(tx["m_s2"], f"{last_obs['date'].iloc[0]:%d.%m.%Y}" if len(last_obs) else "—")

        years = sorted(d_f["date"].dt.year.unique())
        year = day.year if day.year in years else years[-1]
        t1, t2, t3, t4 = st.tabs(tx["tabs"])
        with t1:
            fig = chart_soil(tx, soil_f, year, day) if not soil_f.empty else None
            if fig is None:
                st.info(tx["no_data"])
            else:
                st.plotly_chart(fig, width="stretch")
                st.caption(tx["soil_note"])
        with t2:
            fig = chart_plants(tx, s2f, predf, year, day)
            if fig is None:
                st.info(tx["no_data"])
            else:
                st.plotly_chart(fig, width="stretch")
                st.caption(tx["plant_note"])
        with t3:
            st.plotly_chart(chart_rain(tx, d_f, year), width="stretch")
        with t4:
            an_f = anomalies[anomalies["field"] == field] if not anomalies.empty else anomalies
            fig = chart_anomalies(tx, an_f, day)
            if fig is None:
                st.info(tx["an_none"])
            else:
                now = an_f[(an_f["year"] == day.year) & (an_f["month"] == day.month)]
                month_name = f"{tx['months'][day.month - 1]} {day.year}"
                a1, a2 = st.columns(2)
                a1.metric(tx["an_m_count"], int(an_f["anomaly"].sum()), help=tx["an_m_count_help"])
                a2.metric(tx["an_m_now"].format(m=month_name),
                          f"{now['anomaly_rank'].iloc[0]:.0f} / 100" if len(now) else "—",
                          help=tx["an_m_now_help"])
                st.plotly_chart(fig, width="stretch")
                top = an_f[an_f["anomaly"]].sort_values("anomaly_score", ascending=False)
                if len(top):
                    st.markdown(f"**{tx['an_top']}**")
                    st.markdown("\n".join(
                        f"- **{tx['months'][r.month - 1]} {r.year}** ({r.anomaly_rank:.0f}/100): "
                        f"{describe_drivers(tx, r.drivers)}"
                        for r in top.itertuples()))
                st.caption(tx["an_note"])

    st.divider()
    st.caption(tx["footer"])
