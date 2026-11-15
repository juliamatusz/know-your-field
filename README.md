# Know Your Field

A Streamlit dashboard for farmers in Wielkopolska (Poland): a satellite map of fields, a simple
soil-water status for each field, and charts of soil moisture, crop condition (Sentinel-2),
rainfall and unusual months (Isolation Forest).

Two language versions share one code base:

| File | Purpose |
|---|---|
| `app_farmer_en.py` | English app (entry point) |
| `app_rolnik.py` | Polish app (entry point) |
| `panel.py` | all logic, charts and UI texts (PL/EN) |
| `data/dataset/*.csv` | daily weather, soil moisture and Sentinel-2 data per field |
| `data/model/predykcje.csv` | crop water-stress predictions (optional) |
| `pola/*.geojson` | field boundaries |
| `requirements.txt` | pinned dependencies |
| `.streamlit/config.toml` | theme and server settings |

The app needs no API keys or secrets – it reads only the files in this folder.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_farmer_en.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → choose the repository and branch `main`.
3. Main file path: `app_farmer_en.py` (or `streamlit_app/app_farmer_en.py` if this folder is not the repository root).
4. **Advanced settings** → Python **3.13**.
5. **Deploy**. Add the Polish version as a second app with `app_rolnik.py`.

The app redeploys automatically on every `git push`. On the free tier it sleeps after 12 hours
without visitors – open the link a few minutes before a demo.

## Data sources

NASA POWER / MERRA-2, ERA5-Land (Copernicus), ESA Sentinel-2, NASA SRTM. Basemap: Esri World Imagery.
