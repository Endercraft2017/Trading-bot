#!/usr/bin/env python3
"""
PhantomBot LLM Supervisor
Analyzes trade history using OpenRouter free LLMs,
generates strategy improvement recommendations,
and saves them to a JSON file for the dashboard.

Runs daily via cron. Uses free models — no cost.
Kyle (Ender) / KRMA Security Audit Lab
"""
import sqlite3
import json
import os
import sys
import logging
from datetime import datetime, timedelta

try:
    from openai import OpenAI
except ImportError:
    print("[LLM Supervisor] openai not installed. Run: pip install openai --break-system-packages")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("LLMSupervisor")

DB_PATH           = "/root/.openclaw/workspace/freqtrade-bot/user_data/tradesv3.sqlite"
RECOMMENDATIONS   = "/root/.openclaw/workspace/freqtrade-bot/ml/llm_recommendations.json"
ML_META           = "/root/.openclaw/workspace/freqtrade-bot/ml/models/signal_filter_meta.json"

OPENROUTER_KEY    = os.environ.get("OPENROUTER_KEY", "")
if not OPENROUTER_KEY:
    # Fallback: read from config file if env var not set
    _key_file = "/root/.openclaw/workspace/freqtrade-bot/user_data/openrouter_key.txt"
    import pathlib
    if pathlib.Path(_key_file).exists():
        OPENROUTER_KEY = pathlib.Path(_key_file).read_text().strip()
    else:
        logger.warning("[LLM] OPENROUTER_KEY not set — LLM analysis disabled")
OPENROUTER_BASE   = "https://openrouter.ai/api/v1"

# Free models to try in order
MODEL_CASCADE = [
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-27b-it:free",
]

# Seconds to wait between model attempts to avoid rate limits
LLM_RETRY_DELAY = 8

MIN_TRADES_FOR_ANALYSIS = 3


def load_trade_summary():
    """Load trade statistics from Freqtrade SQLite DB."""
    if not os.path.exists(DB_PATH):
        logger.warning(f"DB not found: {DB_PATH}")
        return None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # Overall stats
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN close_profit <= 0 THEN 1 ELSE 0 END) as losses,
                AVG(close_profit) as avg_profit,
                SUM(close_profit_abs) as total_profit_usdt,
                MAX(close_profit) as best_trade,
                MIN(close_profit) as worst_trade,
                AVG(CASE WHEN close_profit > 0 THEN close_profit END) as avg_win,
                AVG(CASE WHEN close_profit <= 0 THEN close_profit END) as avg_loss
            FROM trades WHERE is_open = 0
        """)
        overall = dict(cursor.fetchone())

        # Per-pair stats
        cursor = conn.execute("""
            SELECT pair,
                COUNT(*) as trades,
                SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) as wins,
                AVG(close_profit) as avg_profit,
                SUM(close_profit_abs) as total_profit
            FROM trades WHERE is_open = 0
            GROUP BY pair ORDER BY total_profit DESC
        """)
        pairs = [dict(r) for r in cursor.fetchall()]

        # Recent 24h trades
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor = conn.execute("""
            SELECT pair, close_profit, close_profit_abs, enter_tag, exit_reason,
                   open_date, close_date
            FROM trades
            WHERE is_open = 0 AND close_date > ?
            ORDER BY close_date DESC LIMIT 50
        """, (yesterday,))
        recent = [dict(r) for r in cursor.fetchall()]

        # Exit reason breakdown
        cursor = conn.execute("""
            SELECT exit_reason, COUNT(*) as count, AVG(close_profit) as avg_profit
            FROM trades WHERE is_open = 0
            GROUP BY exit_reason ORDER BY count DESC
        """)
        exit_reasons = [dict(r) for r in cursor.fetchall()]

        # Hourly performance
        cursor = conn.execute("""
            SELECT strftime('%H', open_date) as hour,
                   COUNT(*) as trades,
                   AVG(close_profit) as avg_profit
            FROM trades WHERE is_open = 0
            GROUP BY hour ORDER BY hour
        """)
        hourly = [dict(r) for r in cursor.fetchall()]

        conn.close()
        return {
            "overall": overall,
            "pairs": pairs[:15],
            "recent_24h": recent,
            "exit_reasons": exit_reasons,
            "hourly_performance": hourly
        }

    except Exception as e:
        logger.error(f"DB error: {e}")
        conn.close()
        return None


def load_ml_meta():
    if os.path.exists(ML_META):
        with open(ML_META) as f:
            return json.load(f)
    return {}


def ask_llm(prompt: str) -> str:
    """Send analysis request to OpenRouter free LLMs."""
    client = OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_BASE)

    for model in MODEL_CASCADE:
        try:
            logger.info(f"[LLM] Trying {model}...")
            resp = client.chat.completions.create(
                model=model,
                max_tokens=2000,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a quantitative trading analyst specializing in cryptocurrency bot strategy optimization.
You analyze trade data and provide specific, actionable recommendations to improve bot performance.
Focus on: win rate improvement, fee reduction, pair selection, timing patterns, and risk management.
Be specific and data-driven. Format recommendations as a JSON object."""
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"[LLM] {model} failed: {str(e)[:100]}")
            import time as _t; _t.sleep(LLM_RETRY_DELAY)
            continue

    return ""


def parse_recommendations(llm_response: str, trade_data: dict) -> dict:
    """Parse LLM response and structure recommendations."""
    # Try to extract JSON from response
    recommendations = {}
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            recommendations = json.loads(json_match.group())
    except Exception:
        pass

    # Always include raw analysis
    overall = trade_data["overall"]
    wins = overall.get("wins", 0) or 0
    total = overall.get("total_trades", 0) or 1
    win_rate = round(wins / total * 100, 1)

    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_trades": overall.get("total_trades", 0),
            "win_rate": win_rate,
            "total_profit_usdt": round(float(overall.get("total_profit_usdt") or 0), 4),
            "avg_profit_per_trade": round(float(overall.get("avg_profit") or 0) * 100, 3),
            "best_trade_pct": round(float(overall.get("best_trade") or 0) * 100, 2),
            "worst_trade_pct": round(float(overall.get("worst_trade") or 0) * 100, 2),
        },
        "top_pairs": trade_data.get("pairs", [])[:5],
        "exit_reasons": trade_data.get("exit_reasons", []),
        "hourly_best": sorted(
            trade_data.get("hourly_performance", []),
            key=lambda x: float(x.get("avg_profit") or 0), reverse=True
        )[:3],
        "llm_analysis": llm_response[:3000] if llm_response else "LLM analysis unavailable",
        "structured": recommendations,
        "status": "ok"
    }


def main():
    logger.info("[LLM Supervisor] Starting analysis run...")

    trade_data = load_trade_summary()
    if not trade_data:
        result = {
            "generated_at": datetime.now().isoformat(),
            "status": "no_data",
            "message": "No trade data yet. Run the bot in paper mode first.",
            "llm_analysis": ""
        }
        with open(RECOMMENDATIONS, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info("[LLM Supervisor] No data — saved empty recommendation")
        return

    overall = trade_data["overall"]
    if (overall.get("total_trades") or 0) < MIN_TRADES_FOR_ANALYSIS:
        result = {
            "generated_at": datetime.now().isoformat(),
            "status": "insufficient_data",
            "message": f"Need at least {MIN_TRADES_FOR_ANALYSIS} closed trades for analysis. Currently: {overall.get('total_trades', 0)}",
            "llm_analysis": ""
        }
        with open(RECOMMENDATIONS, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info("[LLM Supervisor] Insufficient trades")
        return

    ml_meta = load_ml_meta()

    prompt = f"""Analyze this crypto trading bot performance data and provide optimization recommendations.

## Overall Performance
{json.dumps(trade_data['overall'], indent=2)}

## Per-Pair Performance (top 15)
{json.dumps(trade_data['pairs'], indent=2)}

## Recent 24h Trades
{json.dumps(trade_data['recent_24h'][:20], indent=2)}

## Exit Reason Breakdown
{json.dumps(trade_data['exit_reasons'], indent=2)}

## Hourly Performance Pattern
{json.dumps(trade_data['hourly_performance'], indent=2)}

## ML Model Status
{json.dumps(ml_meta, indent=2)}

Based on this data, provide a JSON response with these fields:
{{
  "overall_assessment": "brief assessment of bot health",
  "win_rate_analysis": "why the win rate is what it is",
  "best_pairs": ["pair1", "pair2", "pair3"],
  "worst_pairs": ["pair1", "pair2"],
  "best_trading_hours": [hour1, hour2, hour3],
  "recommended_actions": [
    "specific action 1",
    "specific action 2",
    "specific action 3"
  ],
  "signal_improvements": "suggestions for improving signal quality",
  "risk_adjustments": "any stop loss or take profit adjustments",
  "pairs_to_remove": ["pair to blacklist"],
  "pairs_to_add": ["suggested new pairs to watch"],
  "priority": "high|medium|low"
}}"""

    logger.info("[LLM Supervisor] Requesting analysis from LLM...")
    llm_response = ask_llm(prompt)

    if llm_response:
        logger.info("[LLM Supervisor] Got LLM response, parsing...")
    else:
        logger.warning("[LLM Supervisor] All LLM models failed — saving data only")

    result = parse_recommendations(llm_response, trade_data)

    os.makedirs(os.path.dirname(RECOMMENDATIONS), exist_ok=True)
    with open(RECOMMENDATIONS, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"[LLM Supervisor] Done. Saved to {RECOMMENDATIONS}")
    logger.info(f"[LLM Supervisor] Win rate: {result['summary']['win_rate']}% | "
                f"Total P&L: {result['summary']['total_profit_usdt']} USDT")


if __name__ == "__main__":
    main()
