"""
Tunisia Smart Tourism — Data Collection
  - OpenWeather API (current + forecast)
  - Google Maps Places API (geocoding + place details)
  - Feature Store builder (ETL)
  - Tunisia seed data (regions, locations, holidays)
"""

import os
import logging
from datetime import date, timedelta
from typing import List, Optional

import httpx
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database import (
    Region, Location, PublicHoliday, WeatherSnapshot,
    VisitorRecord, FlightArrival, MLFeatureRow,
    ActivityCategory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_KEY")
GOOGLE_MAPS_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY",  "YOUR_GOOGLE_MAPS_KEY")
OPENWEATHER_BASE     = "https://api.openweathermap.org/data/2.5"
GOOGLE_PLACES_BASE   = "https://maps.googleapis.com/maps/api/place"
GOOGLE_GEOCODE_BASE  = "https://maps.googleapis.com/maps/api/geocode/json"


# ═══════════════════════════════════════════════════════════
#  WEATHER COLLECTOR
# ═══════════════════════════════════════════════════════════

class WeatherCollector:
    """Fetches current + 5-day forecast weather from OpenWeather API."""

    def __init__(self, api_key: str = OPENWEATHER_API_KEY):
        self.api_key = api_key

    def fetch_current(self, lat: float, lon: float) -> Optional[dict]:
        url = f"{OPENWEATHER_BASE}/weather"
        try:
            resp = httpx.get(url, params={
                "lat": lat, "lon": lon,
                "appid": self.api_key, "units": "metric",
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {
                "temperature_avg": data["main"]["temp"],
                "temperature_min": data["main"]["temp_min"],
                "temperature_max": data["main"]["temp_max"],
                "humidity_pct":    data["main"]["humidity"],
                "wind_speed_kmh":  data["wind"]["speed"] * 3.6,
                "rainfall_mm":     data.get("rain", {}).get("1h", 0.0),
                "weather_code":    data["weather"][0]["id"],
                "description":     data["weather"][0]["description"],
            }
        except Exception as e:
            logger.error(f"Weather API error for ({lat},{lon}): {e}")
            return None

    def fetch_forecast_5d(self, lat: float, lon: float) -> List[dict]:
        """Returns daily aggregated forecast for next 5 days."""
        url = f"{OPENWEATHER_BASE}/forecast"
        try:
            resp = httpx.get(url, params={
                "lat": lat, "lon": lon,
                "appid": self.api_key, "units": "metric",
            }, timeout=10)
            resp.raise_for_status()
            items = resp.json().get("list", [])

            daily: dict = {}
            for item in items:
                d = item["dt_txt"][:10]
                if d not in daily:
                    daily[d] = {"temps": [], "rain": 0.0, "code": 800}
                daily[d]["temps"].append(item["main"]["temp"])
                daily[d]["rain"] += item.get("rain", {}).get("3h", 0.0)
                daily[d]["code"]  = item["weather"][0]["id"]

            return [
                {
                    "date":            d,
                    "temperature_avg": round(float(np.mean(v["temps"])), 1),
                    "rainfall_mm":     round(v["rain"], 2),
                    "weather_code":    v["code"],
                }
                for d, v in daily.items()
            ]
        except Exception as e:
            logger.error(f"Forecast API error: {e}")
            return []

    def ingest_for_location(self, db: Session, location_id: str,
                             lat: float, lon: float):
        """Fetch today's weather and persist to weather_snapshots."""
        data = self.fetch_current(lat, lon)
        if not data:
            return
        # Upsert
        existing = db.query(WeatherSnapshot).filter(
            WeatherSnapshot.location_id == location_id,
            WeatherSnapshot.recorded_date == date.today()
        ).first()
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            snap = WeatherSnapshot(
                location_id=location_id,
                recorded_date=date.today(),
                **data,
            )
            db.add(snap)
        db.commit()
        logger.info(f"Weather snapshot saved for location {location_id}")

    def ingest_all_locations(self, db: Session):
        """Fetch weather for every active location in the DB."""
        locations = db.query(Location).filter(Location.is_active == True).all()
        logger.info(f"Fetching weather for {len(locations)} locations...")
        for loc in locations:
            self.ingest_for_location(db, loc.id, loc.latitude, loc.longitude)


# ═══════════════════════════════════════════════════════════
#  GOOGLE MAPS COLLECTOR
# ═══════════════════════════════════════════════════════════

class GoogleMapsCollector:
    """
    Uses Google Maps APIs to:
      - Geocode location names → lat/lon
      - Fetch Place details (rating, photos, opening hours)
      - Enrich Location records in the DB
    """

    def __init__(self, api_key: str = GOOGLE_MAPS_API_KEY):
        self.api_key = api_key

    def geocode(self, address: str) -> Optional[dict]:
        """Return {lat, lon, formatted_address} for a given address string."""
        try:
            resp = httpx.get(GOOGLE_GEOCODE_BASE, params={
                "address": address,
                "key": self.api_key,
            }, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None
            loc = results[0]["geometry"]["location"]
            return {
                "lat": loc["lat"],
                "lon": loc["lng"],
                "formatted_address": results[0]["formatted_address"],
            }
        except Exception as e:
            logger.error(f"Geocode error for '{address}': {e}")
            return None

    def find_place(self, name: str, lat: float, lon: float) -> Optional[str]:
        """
        Search for a place near (lat, lon) by name.
        Returns the Google place_id string, or None.
        """
        try:
            resp = httpx.get(f"{GOOGLE_PLACES_BASE}/findplacefromtext/json", params={
                "input": name,
                "inputtype": "textquery",
                "locationbias": f"circle:5000@{lat},{lon}",
                "fields": "place_id,name",
                "key": self.api_key,
            }, timeout=10)
            resp.raise_for_status()
            candidates = resp.json().get("candidates", [])
            if candidates:
                return candidates[0]["place_id"]
            return None
        except Exception as e:
            logger.error(f"FindPlace error for '{name}': {e}")
            return None

    def get_place_details(self, place_id: str) -> Optional[dict]:
        """
        Fetch detailed info for a place_id.
        Returns dict with rating, user_ratings_total, opening_hours, etc.
        """
        try:
            resp = httpx.get(f"{GOOGLE_PLACES_BASE}/details/json", params={
                "place_id": place_id,
                "fields": "name,rating,user_ratings_total,opening_hours,formatted_address,photos",
                "key": self.api_key,
            }, timeout=10)
            resp.raise_for_status()
            result = resp.json().get("result", {})
            return {
                "name":                result.get("name"),
                "rating":              result.get("rating"),
                "user_ratings_total":  result.get("user_ratings_total"),
                "formatted_address":   result.get("formatted_address"),
                "opening_hours":       result.get("opening_hours", {}).get("weekday_text", []),
                "open_now":            result.get("opening_hours", {}).get("open_now"),
            }
        except Exception as e:
            logger.error(f"PlaceDetails error for '{place_id}': {e}")
            return None

    def enrich_location(self, db: Session, location: Location):
        """
        Find + store the Google place_id for a Location,
        and optionally log place details.
        """
        if location.google_place_id:
            logger.info(f"Location '{location.name_en}' already has place_id, skipping.")
            return

        place_id = self.find_place(location.name_en, location.latitude, location.longitude)
        if place_id:
            location.google_place_id = place_id
            db.commit()
            logger.info(f"  ✅ '{location.name_en}' → place_id: {place_id}")

            details = self.get_place_details(place_id)
            if details:
                logger.info(f"     Rating: {details.get('rating')} "
                            f"({details.get('user_ratings_total')} reviews)")
        else:
            logger.warning(f"  ⚠️  No place found for '{location.name_en}'")

    def enrich_all_locations(self, db: Session):
        """Enrich every active location in the DB with Google Place data."""
        locations = db.query(Location).filter(Location.is_active == True).all()
        logger.info(f"Enriching {len(locations)} locations via Google Maps...")
        for loc in locations:
            self.enrich_location(db, loc)


# ═══════════════════════════════════════════════════════════
#  FEATURE STORE BUILDER
# ═══════════════════════════════════════════════════════════

class FeatureStoreBuilder:
    """
    Builds ml_feature_rows by joining visitor_records + weather_snapshots
    + public_holidays + flight_arrivals, then computes lag/rolling features.
    """

    def __init__(self, db: Session):
        self.db = db

    def _load_base_df(self, location_id: str, start: date, end: date) -> pd.DataFrame:
        from sqlalchemy import text

        all_dates = pd.date_range(start, end, freq="D")
        df = pd.DataFrame({"feature_date": all_dates})
        df["location_id"] = location_id

        # Visitor records
        vr_rows = self.db.execute(
            text("SELECT record_date, visitor_count FROM visitor_records "
                 "WHERE location_id = :lid AND record_date BETWEEN :s AND :e"),
            {"lid": location_id, "s": start, "e": end}
        ).fetchall()
        vr_df = pd.DataFrame(vr_rows, columns=["feature_date", "actual_visitors"])
        vr_df["feature_date"] = pd.to_datetime(vr_df["feature_date"])
        df = df.merge(vr_df, on="feature_date", how="left")

        # Weather snapshots
        ws_rows = self.db.execute(
            text("SELECT recorded_date, temperature_avg, rainfall_mm, "
                 "wind_speed_kmh, weather_code FROM weather_snapshots "
                 "WHERE location_id = :lid AND recorded_date BETWEEN :s AND :e"),
            {"lid": location_id, "s": start, "e": end}
        ).fetchall()
        ws_df = pd.DataFrame(ws_rows, columns=[
            "feature_date", "temperature_avg", "rainfall_mm",
            "wind_speed_kmh", "weather_code"
        ])
        ws_df["feature_date"] = pd.to_datetime(ws_df["feature_date"])
        df = df.merge(ws_df, on="feature_date", how="left")

        # Public holidays
        hol_rows = self.db.execute(
            text("SELECT date, is_school FROM public_holidays WHERE date BETWEEN :s AND :e"),
            {"s": start, "e": end}
        ).fetchall()
        hol_df = pd.DataFrame(hol_rows, columns=["feature_date", "is_school"])
        hol_df["feature_date"]    = pd.to_datetime(hol_df["feature_date"])
        hol_df["is_public_holiday"] = 1
        hol_df["is_school_holiday"] = hol_df["is_school"].astype(int)
        hol_df = hol_df.drop(columns=["is_school"])
        df = df.merge(hol_df, on="feature_date", how="left")
        df["is_public_holiday"] = df["is_public_holiday"].fillna(0).astype(int)
        df["is_school_holiday"] = df["is_school_holiday"].fillna(0).astype(int)

        # Flight arrivals
        fa_rows = self.db.execute(
            text("SELECT arrival_date, SUM(total_passengers) AS total "
                 "FROM flight_arrivals WHERE arrival_date BETWEEN :s AND :e "
                 "GROUP BY arrival_date"),
            {"s": start, "e": end}
        ).fetchall()
        fa_df = pd.DataFrame(fa_rows, columns=["feature_date", "flight_arrivals_nearest"])
        fa_df["feature_date"] = pd.to_datetime(fa_df["feature_date"])
        df = df.merge(fa_df, on="feature_date", how="left")
        df["flight_arrivals_nearest"] = df["flight_arrivals_nearest"].fillna(0).astype(int)

        # Fill missing weather with seasonal approximation
        df["temperature_avg"] = df["temperature_avg"].fillna(
            df["feature_date"].dt.month.map(
                lambda m: 15 + 12 * np.sin((m - 4) * np.pi / 6)
            )
        )
        df["rainfall_mm"]    = df["rainfall_mm"].fillna(0)
        df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(15)
        df["weather_code"]   = df["weather_code"].fillna(800)

        # Placeholder event columns (no events table in simplified version)
        df["event_count_in_region"] = 0
        df["is_major_festival"]     = 0

        return df

    def build_for_location(self, location_id: str, start: date, end: date):
        from pipeline import engineer_features

        df = self._load_base_df(location_id, start, end)
        df = engineer_features(df)

        count = 0
        for _, row in df.iterrows():
            obj = MLFeatureRow(
                location_id              = location_id,
                feature_date             = row["feature_date"].date()
                    if hasattr(row["feature_date"], "date") else row["feature_date"],
                day_of_week              = int(row.get("day_of_week", 0)),
                day_of_month             = int(row.get("day_of_month", 1)),
                month                    = int(row.get("month", 1)),
                quarter                  = int(row.get("quarter", 1)),
                year                     = int(row.get("year", 2024)),
                week_of_year             = int(row.get("week_of_year", 1)),
                is_weekend               = bool(row.get("is_weekend", False)),
                season                   = row.get("season", "summer"),
                is_public_holiday        = bool(row.get("is_public_holiday", False)),
                is_school_holiday        = bool(row.get("is_school_holiday", False)),
                days_to_next_holiday     = int(row.get("days_to_next_holiday", 365)),
                temperature_avg          = float(row.get("temperature_avg") or 22),
                rainfall_mm              = float(row.get("rainfall_mm") or 0),
                wind_speed_kmh           = float(row.get("wind_speed_kmh") or 15),
                weather_code             = int(row.get("weather_code") or 800),
                event_count_in_region    = int(row.get("event_count_in_region") or 0),
                is_major_festival        = bool(row.get("is_major_festival", False)),
                days_to_nearest_event    = int(row.get("days_to_nearest_event") or 30),
                flight_arrivals_nearest  = int(row.get("flight_arrivals_nearest") or 0),
                visitors_lag_1d          = _safe_int(row.get("visitors_lag_1d")),
                visitors_lag_7d          = _safe_int(row.get("visitors_lag_7d")),
                visitors_lag_365d        = _safe_int(row.get("visitors_lag_365d")),
                visitors_rolling_7d_avg  = _safe_float(row.get("visitors_rolling_7d_avg")),
                visitors_rolling_30d_avg = _safe_float(row.get("visitors_rolling_30d_avg")),
                actual_visitors          = _safe_int(row.get("actual_visitors")),
            )
            # Upsert (delete old, insert new)
            self.db.query(MLFeatureRow).filter(
                MLFeatureRow.location_id == location_id,
                MLFeatureRow.feature_date == obj.feature_date
            ).delete()
            self.db.add(obj)
            count += 1

            if count % 100 == 0:
                self.db.commit()

        self.db.commit()
        logger.info(f"Feature store updated for {location_id}: {count} rows ({start} → {end})")

    def build_all(self, days_back: int = 730):
        locations = self.db.query(Location).filter(Location.is_active == True).all()
        end   = date.today()
        start = end - timedelta(days=days_back)
        for loc in locations:
            logger.info(f"Building features for: {loc.name_en}")
            self.build_for_location(loc.id, start, end)


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return int(v)
    except Exception:
        return None


def _safe_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
#  SEED DATA — Tunisia
# ═══════════════════════════════════════════════════════════

TUNISIA_REGIONS = [
    {"id": 1, "name_en": "Tunis",    "name_ar": "تونس",     "name_fr": "Tunis",
     "governorate": "Tunis",         "latitude": 36.8190, "longitude": 10.1658},
    {"id": 2, "name_en": "Sfax",     "name_ar": "صفاقس",    "name_fr": "Sfax",
     "governorate": "Sfax",          "latitude": 34.7406, "longitude": 10.7603},
    {"id": 3, "name_en": "Sousse",   "name_ar": "سوسة",     "name_fr": "Sousse",
     "governorate": "Sousse",        "latitude": 35.8256, "longitude": 10.6369},
    {"id": 4, "name_en": "Kairouan", "name_ar": "القيروان", "name_fr": "Kairouan",
     "governorate": "Kairouan",      "latitude": 35.6781, "longitude": 10.0963},
    {"id": 5, "name_en": "Djerba",   "name_ar": "جربة",     "name_fr": "Djerba",
     "governorate": "Médenine",      "latitude": 33.8075, "longitude": 10.8451},
    {"id": 6, "name_en": "Douz",     "name_ar": "دوز",      "name_fr": "Douz",
     "governorate": "Kébili",        "latitude": 33.4576, "longitude":  9.0224},
    {"id": 7, "name_en": "Hammamet", "name_ar": "الحمامات", "name_fr": "Hammamet",
     "governorate": "Nabeul",        "latitude": 36.3988, "longitude": 10.6112},
    {"id": 8, "name_en": "Bizerte",  "name_ar": "بنزرت",    "name_fr": "Bizerte",
     "governorate": "Bizerte",       "latitude": 37.2746, "longitude":  9.8739},
]

TUNISIA_LOCATIONS = [
    {
        "name_en": "Bardo National Museum", "name_ar": "المتحف الوطني بباردو",
        "region_id": 1, "category": "history",
        "latitude": 36.8088, "longitude": 10.1341,
        "entrance_fee": 11.0, "capacity": 1500,
        "is_museum": True, "has_vr": True, "eco_rating": 4.0,
        "description_en": "World-renowned museum of Roman mosaics and Tunisian antiquities.",
        "tags": ["roman", "mosaic", "unesco", "museum"],
    },
    {
        "name_en": "Amphitheatre of El Jem", "name_ar": "مدرج الجم",
        "region_id": 2, "category": "history",
        "latitude": 35.2967, "longitude": 10.7072,
        "entrance_fee": 12.0, "capacity": 2000,
        "is_museum": False, "has_vr": True, "eco_rating": 3.8,
        "description_en": "Third-largest Roman amphitheatre in the world.",
        "tags": ["roman", "amphitheatre", "unesco"],
    },
    {
        "name_en": "Great Mosque of Kairouan", "name_ar": "جامع عقبة",
        "region_id": 4, "category": "culture",
        "latitude": 35.6836, "longitude": 10.0966,
        "entrance_fee": 5.0, "capacity": 3000,
        "is_museum": False, "has_vr": False, "eco_rating": 4.5,
        "description_en": "One of Islam's oldest and most important mosques.",
        "tags": ["islamic", "heritage", "unesco", "spiritual"],
    },
    {
        "name_en": "Carthage Archaeological Site", "name_ar": "قرطاج الأثرية",
        "region_id": 1, "category": "history",
        "latitude": 36.8581, "longitude": 10.3297,
        "entrance_fee": 12.0, "capacity": 4000,
        "is_museum": False, "has_vr": True, "eco_rating": 4.2,
        "description_en": "UNESCO World Heritage site — ruins of ancient Carthage.",
        "tags": ["carthage", "phoenician", "roman", "unesco"],
    },
    {
        "name_en": "Djerba Beach — Sidi Mahrez", "name_ar": "شاطئ سيدي مهرز",
        "region_id": 5, "category": "beach",
        "latitude": 33.8700, "longitude": 10.8600,
        "entrance_fee": 0.0, "capacity": 5000,
        "is_museum": False, "has_vr": False, "eco_rating": 4.7,
        "description_en": "Crystal-clear waters and white sand on Djerba Island.",
        "tags": ["beach", "swimming", "family", "island"],
    },
    {
        "name_en": "Sahara Desert — Douz Gateway", "name_ar": "بوابة الصحراء دوز",
        "region_id": 6, "category": "desert",
        "latitude": 33.4500, "longitude":  9.0200,
        "entrance_fee": 25.0, "capacity": 500,
        "is_museum": False, "has_vr": False, "eco_rating": 5.0,
        "description_en": "Gateway to the Grand Erg Oriental — camel treks and dune camps.",
        "tags": ["desert", "adventure", "camel", "camping", "sunset"],
    },
    {
        "name_en": "Medina of Tunis", "name_ar": "مدينة تونس العتيقة",
        "region_id": 1, "category": "culture",
        "latitude": 36.7985, "longitude": 10.1703,
        "entrance_fee": 0.0, "capacity": 8000,
        "is_museum": False, "has_vr": False, "eco_rating": 3.5,
        "description_en": "UNESCO-listed historic medina with souks and traditional architecture.",
        "tags": ["medina", "souk", "culture", "food", "artisan"],
    },
    {
        "name_en": "Hammamet Beach Resort Area", "name_ar": "منطقة شواطئ الحمامات",
        "region_id": 7, "category": "beach",
        "latitude": 36.4000, "longitude": 10.6100,
        "entrance_fee": 0.0, "capacity": 10000,
        "is_museum": False, "has_vr": False, "eco_rating": 3.0,
        "description_en": "Tunisia's premier beach resort with golden sandy beaches.",
        "tags": ["beach", "resort", "family", "watersports"],
    },
]

TUNISIA_HOLIDAYS = [
    {"date": date(2024, 1, 1),  "name_en": "New Year's Day",    "is_school": False},
    {"date": date(2024, 3, 20), "name_en": "Independence Day",  "is_school": False},
    {"date": date(2024, 4, 9),  "name_en": "Martyrs' Day",      "is_school": False},
    {"date": date(2024, 5, 1),  "name_en": "Labour Day",        "is_school": False},
    {"date": date(2024, 7, 7),  "name_en": "Eid al-Adha",       "is_school": False},
    {"date": date(2024, 7, 8),  "name_en": "Eid al-Adha (2nd)", "is_school": False},
    {"date": date(2024, 7, 25), "name_en": "Republic Day",      "is_school": False},
    {"date": date(2024, 8, 13), "name_en": "Women's Day",       "is_school": False},
    {"date": date(2024, 10,15), "name_en": "Evacuation Day",    "is_school": False},
    {"date": date(2024, 12,17), "name_en": "Revolution Day",    "is_school": False},
    {"date": date(2025, 1, 1),  "name_en": "New Year's Day",    "is_school": False},
    {"date": date(2025, 3, 20), "name_en": "Independence Day",  "is_school": False},
    # Summer school holidays July + August 2024
    *[{"date": date(2024, m, d), "name_en": "School Summer Holiday", "is_school": True}
      for m in [7, 8] for d in range(1, 32)
      if date(2024, m, 1) <= date(2024, m, d) <= date(2024, m, 28 if m == 2 else 31)],
]


def seed_database(db: Session):
    """Populate the DB with Tunisia regions, locations, and holidays."""
    # Regions
    for r in TUNISIA_REGIONS:
        if not db.query(Region).filter(Region.id == r["id"]).first():
            db.add(Region(**r))
    db.commit()
    logger.info(f"✅ Seeded {len(TUNISIA_REGIONS)} regions")

    # Locations
    count = 0
    for loc_data in TUNISIA_LOCATIONS:
        if not db.query(Location).filter(Location.name_en == loc_data["name_en"]).first():
            db.add(Location(**loc_data))
            count += 1
    db.commit()
    logger.info(f"✅ Seeded {count} locations")

    # Holidays
    count = 0
    for h in TUNISIA_HOLIDAYS:
        if not db.query(PublicHoliday).filter(PublicHoliday.date == h["date"]).first():
            db.add(PublicHoliday(**h))
            count += 1
    db.commit()
    logger.info(f"✅ Seeded {count} holidays")
