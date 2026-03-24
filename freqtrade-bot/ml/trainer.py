#!/usr/bin/env python3
"""
PhantomBot ML Trainer
Reads closed trades from Freqtrade SQLite DB, engineers features,
trains a GradientBoosting classifier, and saves the model.

Run manually or via cron (daily at 3am):
  0 3 * * * /root/.openclaw/workspace/freqtrade-bot/scripts/retrain.sh

Kyle (Ender) / KRMA Security Audit Lab
"""
import sqlite3
import json
import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    print("[TRAINER] scikit-learn not installed. Run: pip install scikit-learn joblib")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Load .env if present
from pathlib import Path as _Path
_env_file = _Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
logger = logging.getLogger("PhantomTrainer")

DB_PATH     = "/root/.openclaw/workspace/freqtrade-bot/user_data/tradesv3.sqlite"
MODEL_PATH  = "/root/.openclaw/workspace/freqtrade-bot/ml/models/signal_filter.pkl"
META_PATH   = "/root/.openclaw/workspace/freqtrade-bot/ml/models/signal_filter_meta.json"
MIN_TRADES  = 10   # Minimum closed trades to train
ARCHIVE_DIR  = "/root/.openclaw/workspace/freqtrade-bot/ml/models/archive"
HISTORY_PATH = "/root/.openclaw/workspace/freqtrade-bot/ml/models/auc_history.json"
PROFIT_TARGET = 0.003  # 0.3% net profit = label 1

FEATURE_COLUMNS = [
    'rsi', 'rsi1', 'mch', 'mcs',
    'es', 'an', 'volr',
    'hod', 'dow', 'bbp',
    'r1', 'r3', 'r6'
]


def load_trades():
    """Load closed trades from Freqtrade SQLite DB."""
    if not os.path.exists(DB_PATH):
        logger.error(f"DB not found: {DB_PATH}")
        return None

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT 
            id, pair, close_profit, close_profit_abs,
            open_date, close_date, open_rate, close_rate,
            amount, stake_amount, fee_open, fee_close,
            enter_tag, exit_reason, is_open
        FROM trades
        WHERE is_open = 0
        ORDER BY close_date ASC
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"[Trainer] Loaded {len(df)} closed trades from DB")
        return df
    except Exception as e:
        logger.error(f"[Trainer] DB query failed: {e}")
        conn.close()
        return None


def engineer_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from trade metadata stored in enter_tag field.
    PhantomAdaptive stores a JSON feature snapshot in enter_tag.
    Falls back to basic trade-level features if snapshot not available.
    """
    records = []
    for _, row in trades_df.iterrows():
        # Try to parse feature snapshot from enter_tag
        features = None
        if row.get('enter_tag') and str(row['enter_tag']).startswith('{'):
            try:
                features = json.loads(row['enter_tag'])
            except Exception:
                pass

        if features:
            # Fill time fields from open_date if not stored in JSON
            if 'hod' not in features or 'dow' not in features:
                open_dt = pd.to_datetime(row['open_date'])
                features.setdefault('hod', float(open_dt.hour))
                features.setdefault('dow', float(open_dt.weekday()))
        if features and all(k in features for k in FEATURE_COLUMNS):
            feat_vec = [features[k] for k in FEATURE_COLUMNS]
        else:
            # Fallback: derive basic features from trade data
            open_dt = pd.to_datetime(row['open_date'])
            feat_vec = [
                50.0,   # rsi placeholder
                50.0,   # rsi1
                0.0,    # mch
                0.0,    # mcs
                0.0,    # es
                float(row.get('an', row.get('atr_norm', 0.002))),
                1.0,    # volr
                float(open_dt.hour),
                float(open_dt.weekday()),
                0.5,    # bbp
                0.0,    # r1
                0.0,    # r3
                0.0,    # r6
            ]

        # Label: 1 if profit > target, 0 if not
        net_profit = float(row['close_profit']) - float(row.get('fee_open', 0.001)) - float(row.get('fee_close', 0.001))
        label = 1 if net_profit >= PROFIT_TARGET else 0

        records.append({
            **dict(zip(FEATURE_COLUMNS, feat_vec)),
            'label': label,
            'profit_ratio': float(row['close_profit']),
            'pair': row['pair'],
            'open_date': row['open_date']
        })

    return pd.DataFrame(records)


def train(features_df: pd.DataFrame):
    """Train GradientBoosting classifier and save model."""
    X = features_df[FEATURE_COLUMNS].values
    y = features_df['label'].values

    n_pos = y.sum()
    n_neg = (1 - y).sum()
    logger.info(f"[Trainer] Class distribution: {n_pos} profitable ({n_pos/len(y)*100:.1f}%), {n_neg} losing")

    if len(X) < MIN_TRADES:
        logger.warning(f"[Trainer] Only {len(X)} trades — need {MIN_TRADES} minimum. Skipping.")
        return None, None

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if n_pos > 1 and n_neg > 1 else None)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.55).astype(int)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = 0.5  # Can't compute AUC with one class
        logger.warning("[Trainer] Only one class in test set — AUC defaulted to 0.5")

    logger.info(f"[Trainer] ROC-AUC: {auc:.4f}")
    logger.info(f"[Trainer] Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"[Trainer] Top features: {top_features}")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_trades": len(features_df),
        "auc": round(auc, 4),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "feature_importances": importances,
        "model_threshold": 0.55,
        "profit_target": PROFIT_TARGET,
        "deploy": auc >= 0.55  # initial; save_model may override after comparison
    }

    return model, meta


def save_model(model, meta):
    import shutil
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # Load previous AUC for comparison
    prev_auc = 0.0
    prev_deploy = False
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH) as f:
                prev_meta = json.load(f)
            prev_auc   = float(prev_meta.get('auc', 0.0))
            prev_deploy = bool(prev_meta.get('deploy', False))
        except Exception:
            pass

    new_auc = meta['auc']
    # Deploy if: AUC >= 0.55 AND new model is not worse than previous by more than 0.01
    should_deploy = new_auc >= 0.55 and new_auc >= (prev_auc - 0.01)
    meta['deploy']       = should_deploy
    meta['prev_auc']     = round(prev_auc, 4)
    meta['auc_delta']    = round(new_auc - prev_auc, 4)

    # Count real-feature trades
    real_n = sum(1 for f in meta.get('feature_importances', {}) if False)  # placeholder, set by caller
    meta.setdefault('real_feature_trades', 0)

    # Append to AUC history
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append({
        "trained_at":  meta["trained_at"],
        "auc":         new_auc,
        "prev_auc":    round(prev_auc, 4),
        "delta":       round(new_auc - prev_auc, 4),
        "n_trades":    meta["n_trades"],
        "real_trades": meta.get("real_feature_trades", 0),
        "deployed":    should_deploy,
    })
    # Keep last 50 entries
    history = history[-50:]
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    # Archive current deployed model before overwriting
    if should_deploy and os.path.exists(MODEL_PATH):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_pkl  = os.path.join(ARCHIVE_DIR, f'signal_filter_{ts}_auc{prev_auc:.4f}.pkl')
        archive_meta = os.path.join(ARCHIVE_DIR, f'signal_filter_{ts}_auc{prev_auc:.4f}_meta.json')
        shutil.copy2(MODEL_PATH, archive_pkl)
        if os.path.exists(META_PATH):
            shutil.copy2(META_PATH, archive_meta)
        # Keep only last 10 archives
        archives = sorted([f for f in os.listdir(ARCHIVE_DIR) if f.endswith('.pkl')])
        for old in archives[:-10]:
            try:
                os.remove(os.path.join(ARCHIVE_DIR, old))
                os.remove(os.path.join(ARCHIVE_DIR, old.replace('.pkl', '_meta.json')))
            except Exception:
                pass
        logger.info(f"[Trainer] Archived previous model → {archive_pkl}")

    # Always write meta (for dashboard)
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    if should_deploy:
        joblib.dump(model, MODEL_PATH)
        logger.info(f"[Trainer] Deployed new model  AUC={new_auc:.4f}  (was {prev_auc:.4f}, delta={new_auc-prev_auc:+.4f})")
    else:
        if new_auc < 0.55:
            logger.warning(f"[Trainer] AUC {new_auc:.4f} < 0.55 — NOT deploying (keeping previous)")
        else:
            logger.warning(f"[Trainer] AUC {new_auc:.4f} is worse than previous {prev_auc:.4f} — NOT deploying (keeping best)")


def main():
    logger.info("[Trainer] === PhantomBot ML Training Run ===")

    trades = load_trades()
    if trades is None or len(trades) == 0:
        logger.warning("[Trainer] No trades to train on. Run the bot in paper mode first.")
        sys.exit(0)

    features = engineer_features(trades)
    logger.info(f"[Trainer] Feature matrix: {features.shape}")

    model, meta = train(features)
    if model is None:
        sys.exit(0)

    # Count real-feature trades (have actual indicator JSON, not placeholder)
    real_count = sum(
        1 for _, row in trades.iterrows()
        if str(row.get('enter_tag', '')).startswith('{')
    )
    meta['real_feature_trades'] = real_count
    logger.info(f"[Trainer] Real-feature trades: {real_count} / {len(trades)} total")

    save_model(model, meta)
    logger.info("[Trainer] Done.")


if __name__ == "__main__":
    main()
