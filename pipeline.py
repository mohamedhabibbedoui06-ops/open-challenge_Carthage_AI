"""
Tunisia Smart Tourism — ML Pipeline
Feature Engineering + XGBoost + LSTM + Ensemble
"""

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./artifacts"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
    "is_weekend", "season_enc",
    "is_public_holiday", "is_school_holiday", "days_to_next_holiday",
    "temperature_avg", "rainfall_mm", "wind_speed_kmh", "weather_code",
    "event_count_in_region", "is_major_festival", "days_to_nearest_event",
    "flight_arrivals_nearest",
    "visitors_lag_1d", "visitors_lag_7d", "visitors_lag_365d",
    "visitors_rolling_7d_avg", "visitors_rolling_30d_avg",
]
TARGET_COLUMN = "actual_visitors"
SEASON_MAP = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}


def get_season(month: int) -> str:
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    if month in (9, 10, 11): return "autumn"
    return "winter"


# ═══════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["feature_date"] = pd.to_datetime(df["feature_date"])
    df = df.sort_values(["location_id", "feature_date"]).reset_index(drop=True)

    # Temporal
    df["day_of_week"]  = df["feature_date"].dt.dayofweek
    df["day_of_month"] = df["feature_date"].dt.day
    df["month"]        = df["feature_date"].dt.month
    df["quarter"]      = df["feature_date"].dt.quarter
    df["week_of_year"] = df["feature_date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["feature_date"].dt.year
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["season"]       = df["month"].apply(get_season)
    df["season_enc"]   = df["season"].map(SEASON_MAP)

    # Holiday proximity
    holiday_dates = df[df["is_public_holiday"] == 1]["feature_date"].unique()

    def days_to_next(d):
        future = [h for h in holiday_dates if h >= d]
        return (min(future) - d).days if future else 365

    df["days_to_next_holiday"] = df["feature_date"].apply(days_to_next)

    # Event proximity (simple proxy)
    df["days_to_nearest_event"] = (
        df.groupby("location_id")["event_count_in_region"]
        .transform(lambda s: s.eq(0).astype(int).cumsum())
        .clip(0, 30)
    )

    # Lag features
    df["visitors_lag_1d"]   = df.groupby("location_id")["actual_visitors"].shift(1)
    df["visitors_lag_7d"]   = df.groupby("location_id")["actual_visitors"].shift(7)
    df["visitors_lag_365d"] = df.groupby("location_id")["actual_visitors"].shift(365)

    # Rolling averages
    df["visitors_rolling_7d_avg"] = (
        df.groupby("location_id")["actual_visitors"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df["visitors_rolling_30d_avg"] = (
        df.groupby("location_id")["actual_visitors"]
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )

    # Fill NaN lags with location median
    for col in ["visitors_lag_1d", "visitors_lag_7d", "visitors_lag_365d",
                "visitors_rolling_7d_avg", "visitors_rolling_30d_avg",
                "temperature_avg", "rainfall_mm", "wind_speed_kmh"]:
        df[col] = df.groupby("location_id")[col].transform(
            lambda s: s.fillna(s.median())
        )

    return df


# ═══════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "mape": round(mape, 2), "r2": round(r2, 4)}


# ═══════════════════════════════════════════════════════════
#  XGBOOST MODEL
# ═══════════════════════════════════════════════════════════

class XGBoostVisitorModel:
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "n_estimators": 500, "max_depth": 7, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
            "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "objective": "reg:squarederror", "tree_method": "hist",
            "random_state": 42, "n_jobs": -1,
        }
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_importance_: Optional[pd.Series] = None

    def train(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        df = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)
        df = df.sort_values("feature_date")
        X = df[FEATURE_COLUMNS].values
        y = df[TARGET_COLUMN].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            model = xgb.XGBRegressor(**self.params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            scores = evaluate(y[val_idx], model.predict(X[val_idx]))
            cv_scores.append(scores)
            logger.info(f"  XGB Fold {fold+1}: MAE={scores['mae']}  RMSE={scores['rmse']}")

        # Final model on all data
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, verbose=False)
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)

        avg = {k: round(float(np.mean([s[k] for s in cv_scores])), 4) for k in cv_scores[0]}
        logger.info(f"XGBoost CV avg: {avg}")
        return avg

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        X = df[FEATURE_COLUMNS].fillna(0).values
        return np.maximum(self.model.predict(X), 0).astype(int)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / "xgboost_visitor_model.pkl")
        joblib.dump({"model": self.model, "params": self.params,
                     "features": FEATURE_COLUMNS}, path)
        logger.info(f"XGBoost saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "XGBoostVisitorModel":
        data = joblib.load(path)
        obj = cls(params=data["params"])
        obj.model = data["model"]
        return obj


# ═══════════════════════════════════════════════════════════
#  LSTM MODEL
# ═══════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len]


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class LSTMVisitorModel:
    def __init__(self, seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, epochs: int = 50,
                 batch_size: int = 64, lr: float = 1e-3):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.scaler_X    = MinMaxScaler()
        self.scaler_y    = MinMaxScaler()
        self.model: Optional[LSTMNet] = None
        self.input_size  = len(FEATURE_COLUMNS)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self):
        self.model = LSTMNet(self.input_size, self.hidden_size, self.num_layers).to(self.device)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        df = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS).sort_values("feature_date")
        X  = self.scaler_X.fit_transform(df[FEATURE_COLUMNS].values.astype(np.float32))
        y  = self.scaler_y.fit_transform(df[TARGET_COLUMN].values.astype(np.float32).reshape(-1,1)).ravel()

        split    = int(len(X) * 0.85)
        train_dl = DataLoader(TimeSeriesDataset(X[:split], y[:split], self.seq_len),
                              batch_size=self.batch_size, shuffle=False)
        val_dl   = DataLoader(TimeSeriesDataset(X[split:], y[split:], self.seq_len),
                              batch_size=self.batch_size, shuffle=False)

        self._build()
        opt  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit = nn.MSELoss()
        best_loss, best_state = float("inf"), None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_dl:
                    val_losses.append(crit(self.model(xb.to(self.device)), yb.to(self.device)).item())
            vl = float(np.mean(val_losses))
            sch.step(vl)
            if vl < best_loss:
                best_loss  = vl
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            if epoch % 10 == 0:
                logger.info(f"  LSTM Epoch {epoch}/{self.epochs}  val_loss={vl:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)

        # Evaluate
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.extend(self.model(xb.to(self.device)).cpu().numpy())
                trues.extend(yb.numpy())

        y_pred = self.scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).ravel()
        y_true = self.scaler_y.inverse_transform(np.array(trues).reshape(-1,1)).ravel()
        metrics = evaluate(y_true, y_pred)
        logger.info(f"LSTM val metrics: {metrics}")
        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        X = self.scaler_X.transform(df.sort_values("feature_date")[FEATURE_COLUMNS].fillna(0).values.astype(np.float32))
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(self.seq_len, len(X) + 1):
                seq = torch.FloatTensor(X[max(0, i-self.seq_len):i]).unsqueeze(0).to(self.device)
                preds.append(self.model(seq).cpu().item())
        padding = [preds[0]] * (len(X) - len(preds))
        preds = padding + preds
        return np.maximum(
            self.scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).ravel(), 0
        ).astype(int)

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / "lstm_visitor_model.pt")
        torch.save({
            "state_dict": self.model.state_dict() if self.model else None,
            "config": {"seq_len": self.seq_len, "hidden_size": self.hidden_size,
                       "num_layers": self.num_layers, "epochs": self.epochs,
                       "batch_size": self.batch_size, "lr": self.lr,
                       "input_size": self.input_size},
            "scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
        }, path)
        logger.info(f"LSTM saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "LSTMVisitorModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        obj  = cls(**{k: v for k, v in cfg.items() if k != "input_size"})
        obj.input_size = cfg["input_size"]
        obj.scaler_X   = ckpt["scaler_X"]
        obj.scaler_y   = ckpt["scaler_y"]
        obj._build()
        if ckpt["state_dict"]:
            obj.model.load_state_dict(ckpt["state_dict"])
        return obj


# ═══════════════════════════════════════════════════════════
#  ENSEMBLE
# ═══════════════════════════════════════════════════════════

class EnsembleVisitorModel:
    def __init__(self, xgb_model: XGBoostVisitorModel,
                 lstm_model: LSTMVisitorModel,
                 xgb_weight: float = 0.6, lstm_weight: float = 0.4):
        self.xgb    = xgb_model
        self.lstm   = lstm_model
        self.w_xgb  = xgb_weight
        self.w_lstm = lstm_weight

    def calibrate_weights(self, holdout_df: pd.DataFrame) -> Tuple[float, float]:
        y_true = holdout_df[TARGET_COLUMN].values
        xp = self.xgb.predict(holdout_df).astype(float)
        lp = self.lstm.predict(holdout_df).astype(float)
        best_mae, best_w = float("inf"), (0.6, 0.4)
        for w in np.arange(0.1, 1.0, 0.1):
            mae = mean_absolute_error(y_true, w * xp + (1-w) * lp)
            if mae < best_mae:
                best_mae = mae
                best_w   = (round(w, 2), round(1-w, 2))
        self.w_xgb, self.w_lstm = best_w
        logger.info(f"Calibrated → XGB={self.w_xgb}  LSTM={self.w_lstm}  MAE={best_mae:.1f}")
        return best_w

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        blend = self.w_xgb * self.xgb.predict(df) + self.w_lstm * self.lstm.predict(df)
        return np.maximum(blend, 0).astype(int)

    def predict_with_interval(self, df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
        preds = self.predict(df).astype(float)
        return pd.DataFrame({
            "predicted_count": preds.astype(int),
            "lower_bound":     (preds * (1 - alpha * 1.5)).astype(int),
            "upper_bound":     (preds * (1 + alpha * 1.5)).astype(int),
        })

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / "ensemble_config.json")
        with open(path, "w") as f:
            json.dump({"w_xgb": self.w_xgb, "w_lstm": self.w_lstm}, f)
        self.xgb.save()
        self.lstm.save()
        return path

    @classmethod
    def load(cls, path: Path) -> "EnsembleVisitorModel":
        with open(path) as f:
            cfg = json.load(f)
        xgb_m  = XGBoostVisitorModel.load(MODELS_DIR / "xgboost_visitor_model.pkl")
        lstm_m = LSTMVisitorModel.load(MODELS_DIR / "lstm_visitor_model.pt")
        return cls(xgb_m, lstm_m, cfg["w_xgb"], cfg["w_lstm"])


# ═══════════════════════════════════════════════════════════
#  TRAINING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

class TrainingOrchestrator:
    """
    Full pipeline: load data → feature engineering → train XGB + LSTM
    → calibrate ensemble → evaluate → save artifacts.
    Uses synthetic data if no CSV provided.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path, parse_dates=["feature_date"])
            logger.info(f"Loaded {len(df)} rows from {self.data_path}")
            return df
        logger.warning("No data_path found — generating synthetic demo data.")
        return self._generate_synthetic_data()

    @staticmethod
    def _generate_synthetic_data(n_locations: int = 5, days: int = 730) -> pd.DataFrame:
        np.random.seed(42)
        records = []
        loc_ids  = [f"loc_{i:03d}" for i in range(n_locations)]
        bases    = [800, 400, 1200, 300, 600]
        start    = date(2022, 1, 1)

        for i, loc_id in enumerate(loc_ids):
            base = bases[i]
            for offset in range(days):
                d     = start + timedelta(days=offset)
                m     = d.month
                smult = 1 + 0.5 * np.sin((m - 4) * np.pi / 6)
                wmult = 1.3 if d.weekday() >= 5 else 1.0
                v     = int(max(0, base * smult * wmult * (1 + np.random.normal(0, 0.1))))
                records.append({
                    "location_id":            loc_id,
                    "feature_date":           d,
                    "actual_visitors":        v,
                    "temperature_avg":        20 + 12*np.sin((m-4)*np.pi/6) + np.random.normal(0,2),
                    "rainfall_mm":            max(0, np.random.exponential(2) if m in [11,12,1,2] else 0),
                    "wind_speed_kmh":         np.random.uniform(5, 30),
                    "weather_code":           800,
                    "is_public_holiday":      int(m==3 and d.day==20),
                    "is_school_holiday":      int(m in [7,8]),
                    "event_count_in_region":  np.random.poisson(0.3),
                    "is_major_festival":      int(m==7 and 15<=d.day<=25),
                    "flight_arrivals_nearest":int(np.random.normal(1500, 200)),
                })

        return pd.DataFrame(records)

    def run(self) -> Dict:
        logger.info("═══ Starting Training Pipeline ═══")

        df_raw = self.load_data()
        logger.info("Engineering features...")
        df = engineer_features(df_raw)

        cutoff     = df["feature_date"].max() - pd.Timedelta(days=60)
        train_df   = df[df["feature_date"] <= cutoff]
        holdout_df = df[df["feature_date"] > cutoff]
        logger.info(f"Train: {len(train_df)} rows  |  Holdout: {len(holdout_df)} rows")

        logger.info("Training XGBoost...")
        xgb_model  = XGBoostVisitorModel()
        xgb_cv     = xgb_model.train(train_df)

        logger.info("Training LSTM...")
        lstm_model  = LSTMVisitorModel(epochs=30)
        lstm_metrics = lstm_model.train(train_df)

        logger.info("Calibrating ensemble weights...")
        ensemble = EnsembleVisitorModel(xgb_model, lstm_model)
        ensemble.calibrate_weights(holdout_df)

        y_true    = holdout_df[TARGET_COLUMN].values
        final     = evaluate(y_true, ensemble.predict(holdout_df))
        logger.info(f"FINAL ENSEMBLE holdout: {final}")

        ens_path = ensemble.save()
        logger.info(f"✅ Artifacts saved to {MODELS_DIR}")

        return {
            "xgb_cv_metrics":   xgb_cv,
            "lstm_val_metrics": lstm_metrics,
            "ensemble_holdout": final,
            "ensemble_weights": {"xgb": ensemble.w_xgb, "lstm": ensemble.w_lstm},
            "artifacts":        str(MODELS_DIR),
        }


# ═══════════════════════════════════════════════════════════
#  INFERENCE SERVICE
# ═══════════════════════════════════════════════════════════

class VisitorPredictionService:
    """Load trained ensemble and run predictions."""

    _instance: Optional["VisitorPredictionService"] = None

    def __init__(self):
        cfg_path = MODELS_DIR / "ensemble_config.json"
        if not cfg_path.exists():
            logger.info("No saved model — training now...")
            TrainingOrchestrator().run()
        self.ensemble = EnsembleVisitorModel.load(cfg_path)
        logger.info("✅ Ensemble model loaded for inference.")

    @classmethod
    def get(cls) -> "VisitorPredictionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_for_location(
        self,
        location_id: str,
        target_dates: List[date],
        context_df: pd.DataFrame,
    ) -> pd.DataFrame:
        loc_df = context_df[context_df["location_id"] == location_id].copy()
        eng    = engineer_features(loc_df)
        target_strs = {str(d) for d in target_dates}
        pred_df = eng[eng["feature_date"].astype(str).isin(target_strs)].copy()
        if pred_df.empty:
            return pd.DataFrame(columns=["feature_date", "predicted_count",
                                         "lower_bound", "upper_bound"])
        intervals = self.ensemble.predict_with_interval(pred_df)
        return pd.concat([pred_df[["feature_date"]].reset_index(drop=True),
                          intervals.reset_index(drop=True)], axis=1)

    def predict_weekly(self, location_id: str, week_start: date,
                       context_df: pd.DataFrame) -> Dict:
        dates = [week_start + timedelta(days=i) for i in range(7)]
        daily = self.predict_for_location(location_id, dates, context_df)
        return {
            "location_id":        location_id,
            "week_start":         str(week_start),
            "daily_predictions":  daily.to_dict(orient="records"),
            "weekly_total":       int(daily["predicted_count"].sum()),
        }
