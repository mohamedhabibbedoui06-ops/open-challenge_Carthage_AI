"""
Tunisia Smart Tourism — Database Models (SQLite, local)
"""

import uuid
import enum
from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    DateTime, Date, Text, ForeignKey, Enum, JSON,
    UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session, sessionmaker

DATABASE_URL = "sqlite:///./tourism.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def get_db() -> Session: # type: ignore
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    pass


# ── Enums ────────────────────────────────────────────────────────────────────

class ActivityCategory(str, enum.Enum):
    BEACH     = "beach"
    HISTORY   = "history"
    DESERT    = "desert"
    FESTIVAL  = "festival"
    FOOD      = "food"
    ADVENTURE = "adventure"
    CULTURE   = "culture"
    NATURE    = "nature"
    WELLNESS  = "wellness"


class Season(str, enum.Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


# ── Models ───────────────────────────────────────────────────────────────────

class Region(Base):
    __tablename__ = "regions"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    name_en     = Column(String(100), nullable=False)
    name_ar     = Column(String(100))
    name_fr     = Column(String(100))
    governorate = Column(String(100))
    latitude    = Column(Float, nullable=False)
    longitude   = Column(Float, nullable=False)

    locations   = relationship("Location", back_populates="region")


class Location(Base):
    __tablename__ = "locations"
    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id       = Column(Integer, ForeignKey("regions.id"), nullable=False)
    name_en         = Column(String(255), nullable=False)
    name_ar         = Column(String(255))
    description_en  = Column(Text)
    category        = Column(String(50), nullable=False)
    latitude        = Column(Float, nullable=False)
    longitude       = Column(Float, nullable=False)
    entrance_fee    = Column(Float, default=0.0)
    capacity        = Column(Integer)
    is_museum       = Column(Boolean, default=False)
    has_vr          = Column(Boolean, default=False)
    eco_rating      = Column(Float)
    tags            = Column(JSON)
    google_place_id = Column(String(255))
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    region            = relationship("Region", back_populates="locations")
    visitor_records   = relationship("VisitorRecord", back_populates="location")
    weather_snapshots = relationship("WeatherSnapshot", back_populates="location")
    predictions       = relationship("VisitorPrediction", back_populates="location")


class WeatherSnapshot(Base):
    __tablename__ = "weather_snapshots"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    location_id     = Column(String(36), ForeignKey("locations.id"), nullable=False)
    recorded_date   = Column(Date, nullable=False)
    temperature_avg = Column(Float)
    temperature_min = Column(Float)
    temperature_max = Column(Float)
    rainfall_mm     = Column(Float, default=0.0)
    wind_speed_kmh  = Column(Float)
    humidity_pct    = Column(Float)
    weather_code    = Column(Integer)
    description     = Column(String(100))

    location = relationship("Location", back_populates="weather_snapshots")

    __table_args__ = (
        UniqueConstraint("location_id", "recorded_date", name="uq_weather_location_date"),
    )


class PublicHoliday(Base):
    __tablename__ = "public_holidays"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    date      = Column(Date, nullable=False, unique=True)
    name_en   = Column(String(255))
    is_school = Column(Boolean, default=False)
    country   = Column(String(10), default="TN")


class VisitorRecord(Base):
    __tablename__ = "visitor_records"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    location_id   = Column(String(36), ForeignKey("locations.id"), nullable=False)
    record_date   = Column(Date, nullable=False)
    visitor_count = Column(Integer, nullable=False)
    source        = Column(String(100))
    created_at    = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="visitor_records")

    __table_args__ = (
        UniqueConstraint("location_id", "record_date", name="uq_visitor_location_date"),
    )


class FlightArrival(Base):
    __tablename__ = "flight_arrivals"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    airport_code     = Column(String(10), nullable=False)
    arrival_date     = Column(Date, nullable=False)
    international    = Column(Integer, default=0)
    domestic         = Column(Integer, default=0)
    total_passengers = Column(Integer)

    __table_args__ = (
        UniqueConstraint("airport_code", "arrival_date", name="uq_flight_airport_date"),
    )


class MLFeatureRow(Base):
    __tablename__ = "ml_feature_rows"
    id                       = Column(Integer, primary_key=True, autoincrement=True)
    location_id              = Column(String(36), ForeignKey("locations.id"), nullable=False)
    feature_date             = Column(Date, nullable=False)
    day_of_week              = Column(Integer)
    day_of_month             = Column(Integer)
    month                    = Column(Integer)
    quarter                  = Column(Integer)
    year                     = Column(Integer)
    week_of_year             = Column(Integer)
    is_weekend               = Column(Boolean, default=False)
    season                   = Column(String(20))
    is_public_holiday        = Column(Boolean, default=False)
    is_school_holiday        = Column(Boolean, default=False)
    days_to_next_holiday     = Column(Integer)
    temperature_avg          = Column(Float)
    rainfall_mm              = Column(Float, default=0.0)
    wind_speed_kmh           = Column(Float)
    weather_code             = Column(Integer)
    event_count_in_region    = Column(Integer, default=0)
    is_major_festival        = Column(Boolean, default=False)
    days_to_nearest_event    = Column(Integer)
    flight_arrivals_nearest  = Column(Integer, default=0)
    visitors_lag_1d          = Column(Integer)
    visitors_lag_7d          = Column(Integer)
    visitors_lag_365d        = Column(Integer)
    visitors_rolling_7d_avg  = Column(Float)
    visitors_rolling_30d_avg = Column(Float)
    actual_visitors          = Column(Integer, nullable=True)
    computed_at              = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("location_id", "feature_date", name="uq_ml_feature_location_date"),
    )


class VisitorPrediction(Base):
    __tablename__ = "visitor_predictions"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    location_id     = Column(String(36), ForeignKey("locations.id"), nullable=False)
    prediction_date = Column(Date, nullable=False)
    predicted_count = Column(Integer, nullable=False)
    lower_bound     = Column(Integer)
    upper_bound     = Column(Integer)
    model_name      = Column(String(100))
    confidence_score= Column(Float)
    generated_at    = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="predictions")


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created (SQLite: tourism.db)")
