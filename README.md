# Tunisia Smart Tourism — Local Setup

A clean, self-contained version of the platform. No Docker, no Redis, no Celery. Just Python + SQLite.

# 🎥 Project Demo Video

❗❗❗
👉 https://drive.google.com/file/d/16niQtdvZhS8kyX-2X_nA2rfTXs74RtAr/view?usp=sharing
❗❗❗

## Project Structure

```
tourism_local/
├── database.py          # SQLAlchemy models (SQLite)
├── data_collection.py   # Weather API + Google Maps API + Tunisia seed data
├── pipeline.py          # Feature engineering + XGBoost + LSTM + Ensemble
├── run.py               # Main entry point
├── requirements.txt
├── .env.example         # Copy to .env and add your API keys
└── artifacts/           # Saved model files (auto-created)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (optional — works without them using synthetic data)
cp .env.example .env
# Edit .env with your keys

# 3. Run everything
python run.py
```

## What It Does

1. **Database** — Creates `tourism.db` (SQLite) with all tables
2. **Seed Data** — Populates 8 Tunisian regions, 8 landmark locations, holidays
3. **Weather** — Fetches live weather for each location via OpenWeather API *(requires key)*
4. **Google Maps** — Enriches locations with Google Place IDs + ratings *(requires key)*
5. **Feature Store** — Builds ML-ready feature rows (lag features, rolling averages, etc.)
6. **ML Training** — Trains XGBoost + LSTM + Ensemble with time-series cross-validation
7. **Predictions** — Prints a sample weekly visitor forecast

## Run Options

```bash
python run.py                        # Full pipeline
python run.py --skip-weather         # Skip weather API calls
python run.py --skip-maps            # Skip Google Maps calls
python run.py --skip-training        # DB + APIs only, no ML
python run.py --predict-only         # Just run predictions (model must exist)
python run.py --data-csv my_data.csv # Use your own historical visitor CSV
```

## CSV Format (for `--data-csv`)

If you have real historical visitor data, provide a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| location_id | string | Unique location identifier |
| feature_date | date | YYYY-MM-DD |
| actual_visitors | int | Visitor count that day |
| temperature_avg | float | °C |
| rainfall_mm | float | mm |
| wind_speed_kmh | float | km/h |
| weather_code | int | OpenWeather code |
| is_public_holiday | int | 0 or 1 |
| is_school_holiday | int | 0 or 1 |
| event_count_in_region | int | Events that day |
| is_major_festival | int | 0 or 1 |
| flight_arrivals_nearest | int | Airport passengers |

## API Keys

**OpenWeather** (free tier, 1000 calls/day):  
https://openweathermap.org/api

**Google Maps** (enable Places API + Geocoding API):  
https://console.cloud.google.com/

Both are optional — the platform runs fully offline using synthetic demo data.
