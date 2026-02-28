"""
Tunisia Smart Tourism — Local Runner
=====================================
Run this file to:
  1. Initialize the SQLite database
  2. Seed Tunisia locations, regions, holidays
  3. Optionally fetch live weather (requires OPENWEATHER_API_KEY)
  4. Optionally enrich locations via Google Maps (requires GOOGLE_MAPS_API_KEY)
  5. Build the ML feature store
  6. Train XGBoost + LSTM + Ensemble models
  7. Run a sample prediction

Usage:
    python run.py                         # Full pipeline with synthetic data
    python run.py --skip-training         # DB + weather only (no ML)
    python run.py --predict-only          # Just run predictions (model must exist)

Set your API keys in a .env file or as environment variables:
    OPENWEATHER_API_KEY=your_key
    GOOGLE_MAPS_API_KEY=your_key
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Tunisia Smart Tourism Local Runner")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip ML training (DB + data collection only)")
    parser.add_argument("--predict-only",   action="store_true",
                        help="Only run predictions using saved model")
    parser.add_argument("--skip-weather",   action="store_true",
                        help="Skip live weather API calls")
    parser.add_argument("--skip-maps",      action="store_true",
                        help="Skip Google Maps enrichment")
    parser.add_argument("--data-csv",       type=str, default=None,
                        help="Path to CSV with historical visitor data (optional)")
    args = parser.parse_args()

    # ── 1. Database setup ──────────────────────────────────────────────────
    logger.info("━━━ Step 1: Database Setup ━━━")
    from database import init_db, SessionLocal
    init_db()
    db = SessionLocal()

    # ── 2. Seed data ───────────────────────────────────────────────────────
    logger.info("━━━ Step 2: Seeding Tunisia Data ━━━")
    from data_collection import seed_database
    seed_database(db)

    if args.predict_only:
        _run_predictions(db)
        db.close()
        return

    # ── 3. Weather collection ──────────────────────────────────────────────
    if not args.skip_weather:
        ow_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not ow_key or ow_key == "YOUR_OPENWEATHER_KEY":
            logger.warning("⚠️  OPENWEATHER_API_KEY not set — skipping weather fetch.\n"
                           "    Set it in .env or environment to enable live weather data.")
        else:
            logger.info("━━━ Step 3: Fetching Live Weather ━━━")
            from data_collection import WeatherCollector
            wc = WeatherCollector(api_key=ow_key)
            wc.ingest_all_locations(db)

    # ── 4. Google Maps enrichment ──────────────────────────────────────────
    if not args.skip_maps:
        gm_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        if not gm_key or gm_key == "YOUR_GOOGLE_MAPS_KEY":
            logger.warning("⚠️  GOOGLE_MAPS_API_KEY not set — skipping Maps enrichment.\n"
                           "    Set it in .env or environment to enable Google Places data.")
        else:
            logger.info("━━━ Step 4: Google Maps Enrichment ━━━")
            from data_collection import GoogleMapsCollector
            gmc = GoogleMapsCollector(api_key=gm_key)
            gmc.enrich_all_locations(db)

    # ── 5. Feature store ───────────────────────────────────────────────────
    if not args.skip_training:
        logger.info("━━━ Step 5: Building Feature Store ━━━")
        from data_collection import FeatureStoreBuilder
        fsb = FeatureStoreBuilder(db)
        fsb.build_all(days_back=730)

    # ── 6. Train ML models ────────────────────────────────────────────────
    if not args.skip_training:
        logger.info("━━━ Step 6: Training ML Models ━━━")
        from pipeline import TrainingOrchestrator
        orch    = TrainingOrchestrator(data_path=args.data_csv)
        results = orch.run()
        print("\n" + "="*50)
        print("  TRAINING RESULTS")
        print("="*50)
        print(f"  XGBoost CV:        {results['xgb_cv_metrics']}")
        print(f"  LSTM Validation:   {results['lstm_val_metrics']}")
        print(f"  Ensemble Holdout:  {results['ensemble_holdout']}")
        print(f"  Weights:           {results['ensemble_weights']}")
        print(f"  Artifacts saved:   {results['artifacts']}")
        print("="*50 + "\n")

    # ── 7. Sample predictions ─────────────────────────────────────────────
    _run_predictions(db)

    db.close()
    logger.info("✅ Done!")


def _run_predictions(db):
    logger.info("━━━ Step 7: Sample Predictions ━━━")
    from pipeline import VisitorPredictionService, TrainingOrchestrator

    svc = VisitorPredictionService.get()

    # Use synthetic data as context for demo predictions
    context_df = TrainingOrchestrator._generate_synthetic_data(n_locations=3, days=400)

    loc_id     = "loc_000"
    week_start = date.today()
    result     = svc.predict_weekly(loc_id, week_start, context_df)

    print("\n" + "="*50)
    print("  SAMPLE WEEKLY PREDICTION")
    print("="*50)
    print(f"  Location:      {result['location_id']}")
    print(f"  Week starting: {result['week_start']}")
    print(f"  Weekly total:  {result['weekly_total']:,} visitors\n")
    print(f"  {'Date':<14} {'Predicted':>12}  {'Low':>8}  {'High':>8}")
    print(f"  {'-'*46}")
    for row in result["daily_predictions"]:
        d  = str(row.get("feature_date", ""))[:10]
        pc = row.get("predicted_count", 0)
        lb = row.get("lower_bound", 0)
        ub = row.get("upper_bound", 0)
        print(f"  {d:<14} {pc:>12,}  {lb:>8,}  {ub:>8,}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
