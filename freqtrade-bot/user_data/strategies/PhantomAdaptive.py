"""
PhantomAdaptive - ML-enhanced adaptive layer on top of PhantomStrategy
Uses a trained GradientBoosting classifier to filter/confirm signals.
Falls back to base strategy if model is not trained yet.
Kyle (Ender) / KRMA Security Audit Lab
"""
from pandas import DataFrame
from pathlib import Path
import os
import json
import logging
from datetime import datetime
from typing import Optional
from PhantomStrategy import PhantomStrategy

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

logger = logging.getLogger(__name__)

# Derive paths relative to this file's location (strategies/ -> ml/models/)
_STRATEGY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _STRATEGY_DIR.parent.parent
MODEL_PATH = str(_PROJECT_ROOT / "ml" / "models" / "signal_filter.pkl")
MODEL_META_PATH = str(_PROJECT_ROOT / "ml" / "models" / "signal_filter_meta.json")

# Minimum confidence threshold to confirm a trade entry
MODEL_THRESHOLD = 0.55
# Reload model every N candles
MODEL_RELOAD_INTERVAL = 288  # ~24h at 5m timeframe


class PhantomAdaptive(PhantomStrategy):
    """
    Extends PhantomStrategy with an ML signal filter.
    On every buy signal, a trained GradientBoosting classifier votes
    on whether the trade is likely to be profitable. If the model
    is not yet trained (< 30 closed trades), the base signals pass through.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = None
        self.model_loaded_at: Optional[float] = None
        self._candle_count = 0
        self._load_model()

    def _load_model(self):
        """Load the ML model from disk if available."""
        if not HAS_JOBLIB:
            logger.warning("[PhantomAdaptive] joblib not installed, ML layer disabled")
            return
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.model_loaded_at = os.path.getmtime(MODEL_PATH)
                meta = {}
                if os.path.exists(MODEL_META_PATH):
                    with open(MODEL_META_PATH) as f:
                        meta = json.load(f)
                logger.info(f"[PhantomAdaptive] Model loaded. AUC={meta.get('auc', 'unknown')}, "
                            f"trained on {meta.get('n_trades', '?')} trades")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.error(f"[PhantomAdaptive] Failed to load model: {e}")
                self.model = None
        else:
            logger.info("[PhantomAdaptive] No model found -- running base strategy only")

    def _maybe_reload_model(self):
        """Reload model if file was updated since last load."""
        self._candle_count += 1
        if self._candle_count % MODEL_RELOAD_INTERVAL != 0:
            return
        if not os.path.exists(MODEL_PATH):
            return
        mtime = os.path.getmtime(MODEL_PATH)
        if self.model_loaded_at is None or mtime > self.model_loaded_at:
            logger.info("[PhantomAdaptive] Model file updated -- reloading")
            self._load_model()

    def _extract_features(self, dataframe: DataFrame, metadata: dict) -> Optional[list]:
        """Extract feature vector from latest candle for model prediction.

        NOTE: These features are calculated inline from the dataframe. The trainer
        extracts features from the enter_tag JSON stored at trade entry time.
        Both sources should produce equivalent values, but minor floating-point
        differences may exist. This is a known limitation.
        """
        if len(dataframe) < 8:  # Need at least iloc[-7] access
            return None
        try:
            last = dataframe.iloc[-1]
            prev = dataframe.iloc[-2]
            prev3 = dataframe.iloc[-4]  # 3 candles back from current

            features = [
                float(last.get('rsi', 50)),
                float(prev.get('rsi', 50)),  # RSI lag 1
                float(last.get('macdhist', 0)),
                float(last.get('macdhist', 0) - prev.get('macdhist', 0)),  # MACD slope
                float((last.get('ema_fast', last['close']) - last.get('ema_slow', last['close'])) / last['close']),
                float(last.get('atr', 0) / last['close']),  # normalized ATR
                float(last.get('volume', 0) / max(last.get('volume_ma', 1), 1)),  # volume ratio
                float(last.get('bb_pct', 0.5)),
                float(last.get('date').hour if hasattr(last.get('date'), 'hour') else 0),  # hour of day
                float(last.get('date').weekday() if hasattr(last.get('date'), 'weekday') else 0),  # day of week
                float((last['close'] - prev['close']) / prev['close']),       # 1-candle return
                float((last['close'] - prev3['close']) / prev3['close']),     # 3-candle return
                float((last['close'] - dataframe.iloc[-7]['close']) / dataframe.iloc[-7]['close']),  # 6-candle return
            ]
            return features
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"[PhantomAdaptive] Feature extraction failed: {e}")
            return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                             rate: float, time_in_force: str, current_time,
                             entry_tag: Optional[str] = None, side: str = 'long', **kwargs) -> bool:
        """ML gate: confirm or block a buy signal. Falls back to parent validation if no model."""
        # Always run parent validation first (NaN checks, min stake, EV gate, etc.)
        if not super().confirm_trade_entry(pair, order_type, amount, rate,
                                           time_in_force, current_time,
                                           entry_tag=entry_tag, side=side, **kwargs):
            return False

        if self.model is None:
            return True  # No model -- parent approved, pass through

        # Get the dataframe for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 8:
            return True

        features = self._extract_features(dataframe, {'pair': pair})
        if features is None:
            return True  # Can't extract features -- allow trade

        try:
            proba = self.model.predict_proba([features])[0][1]
            decision = proba >= MODEL_THRESHOLD
            logger.info(f"[PhantomAdaptive] {pair} ML confidence: {proba:.3f} -> {'ALLOW' if decision else 'BLOCK'}")
            return decision
        except (ValueError, IndexError, AttributeError) as e:
            logger.error(f"[PhantomAdaptive] Model prediction failed: {e}")
            return True  # Fail safe: allow trade

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._maybe_reload_model()
        return super().populate_indicators(dataframe, metadata)
