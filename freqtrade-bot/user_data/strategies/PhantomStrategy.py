"""
PhantomStrategy v2 - Full Math Engine
Signal scoring (0-6), ATR-based stops, tiered TP, EV check
Dynamic position sizing, no daily trade limit
Kyle (Ender) / KRMA Security Audit Lab
"""
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import json
import logging

try:
    from freqtrade.persistence import Trade
except ImportError:
    Trade = None

logger = logging.getLogger(__name__)

# Fee per round trip (Binance standard, update to 0.0015 if using BNB)
FEE_ROUNDTRIP = 0.002  # 0.1%x2 standard; use 0.0015 if paying fees in BNB

# Minimum signal score to enter (out of 6)
MIN_SIGNAL_SCORE = 4  # Raised from 3 -- quality over quantity

# ATR multipliers
ATR_STOP_MULT   = 1.5   # stop = entry - (1.5 x ATR)
ATR_TP1_MULT    = 3.0   # TP1  = entry + (3.0 x ATR) -- 2:1 R:R vs 1.5x stop
# Remaining 20% trails via trailing_stop


# Dynamic trade sizing
MIN_STAKE_PER_TRADE = 10.0   # USDT -- minimum per trade slot (clears Binance minimums)
MAX_OPEN_TRADES_CAP = 10     # hard cap -- never open more than this regardless of wallet size
WIN_RATE_ESTIMATE = 0.55  # Conservative estimate for EV calculation


class PhantomStrategy(IStrategy):
    """
    Signal-gated strategy with full math engine.
    Watches up to 30 pairs dynamically, trades only on high-confidence signals.
    """

    INTERFACE_VERSION = 3
    can_short = False
    use_custom_stoploss = False  # ATR stop too tight for 1m candle noise -- use hard stoploss + trailing

    # -- ROI: tightened fallback -- ATR custom_exit handles primary TPs --
    minimal_roi = {
        "180": 0.008,   # fallback: 0.8% after 3h
        "90":  0.015,   # 1.5% after 90min
        "0":   0.05     # immediate exit only if 5%+ (very fast spike)
    }

    # -- Trailing stop locks in profit after TP1 hit --
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    timeframe = "1m"
    startup_candle_count = 200

    # -- Hyperopt parameters --
    buy_ema_fast    = IntParameter(5,  15,  default=9,   space="buy", load=True)
    buy_ema_slow    = IntParameter(18, 35,  default=21,  space="buy", load=True)
    buy_ema_trend   = IntParameter(20, 100, default=50, space="buy", load=True)
    buy_rsi_min     = IntParameter(20, 40,  default=25,  space="buy", load=True)
    buy_rsi_max     = IntParameter(60, 80,  default=72,  space="buy", load=True)
    buy_vol_mult    = DecimalParameter(1.0, 2.0, default=1.2, space="buy", load=True)
    buy_atr_stop    = DecimalParameter(1.0, 2.5, default=1.5, space="buy", load=True)
    buy_atr_tp      = DecimalParameter(2.0, 4.0, default=3.0, space="buy", load=True)

    # Cached recent trades (refreshed each bot loop cycle, not per candle)
    _cached_win_rate: float = WIN_RATE_ESTIMATE

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- Trend EMAs --
        dataframe['ema_fast']  = ta.EMA(dataframe, timeperiod=self.buy_ema_fast.value)
        dataframe['ema_slow']  = ta.EMA(dataframe, timeperiod=self.buy_ema_slow.value)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.buy_ema_trend.value)

        # -- RSI --
        dataframe['rsi']    = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_7']  = ta.RSI(dataframe, timeperiod=7)

        # -- MACD --
        macd = ta.MACD(dataframe)
        dataframe['macd']      = macd['macd']
        dataframe['macdsignal']= macd['macdsignal']
        dataframe['macdhist']  = macd['macdhist']
        dataframe['macd_slope']= dataframe['macdhist'] - dataframe['macdhist'].shift(1)

        # -- Bollinger Bands --
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lower'] = bb['lower']
        dataframe['bb_mid']   = bb['mid']
        dataframe['bb_upper'] = bb['upper']
        dataframe['bb_pct']   = (dataframe['close'] - dataframe['bb_lower']) / (
                                  dataframe['bb_upper'] - dataframe['bb_lower'] + 1e-10)

        # -- ATR (volatility-based stop/target) --
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']

        # -- Volume --
        dataframe['volume_ma']  = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / (dataframe['volume_ma'] + 1e-10)

        # -- Lagged / return features for ML training --
        dataframe['rsi_lag1'] = dataframe['rsi'].shift(1)
        dataframe['ret_1c']   = dataframe['close'].pct_change(1)
        dataframe['ret_3c']   = dataframe['close'].pct_change(3)
        dataframe['ret_6c']   = dataframe['close'].pct_change(6)
        dataframe['ema_spread'] = (dataframe['ema_fast'] - dataframe['ema_slow']) / (dataframe['ema_slow'] + 1e-10)

        # -- ADX (trend strength) --
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # -- Stochastic --
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # -- Signal Score (0-6) --
        score = (
            # 1. EMA fast above slow (trend direction)
            (dataframe['ema_fast'] > dataframe['ema_slow']).astype(int) +
            # 2. Price above 200 EMA (macro trend aligned)
            (dataframe['close'] > dataframe['ema_trend']).astype(int) +
            # 3. RSI in healthy buy zone
            ((dataframe['rsi'] > self.buy_rsi_min.value) &
             (dataframe['rsi'] < self.buy_rsi_max.value)).astype(int) +
            # 4. MACD histogram positive and rising
            ((dataframe['macdhist'] > 0) &
             (dataframe['macd_slope'] > 0)).astype(int) +
            # 5. Volume spike confirmation
            (dataframe['volume_ratio'] > self.buy_vol_mult.value).astype(int) +
            # 6. Not near top of Bollinger Band (room to move up)
            (dataframe['bb_pct'] < 0.75).astype(int)
        )
        dataframe['signal_score'] = score

        # -- EV Check -- use cached win rate from bot_loop_start --
        win_rate = self._cached_win_rate

        tp_pct   = dataframe['atr_pct'] * self.buy_atr_tp.value
        stop_pct = dataframe['atr_pct'] * self.buy_atr_stop.value
        # Include slippage estimate (0.1% on entry + 0.1% on exit for alts)
        slippage_est = 0.001
        dataframe['ev'] = (
            win_rate * tp_pct -
            (1 - win_rate) * stop_pct -
            FEE_ROUNDTRIP -
            slippage_est
        )

        # -- EMA just crossed (fresh signal, not stale) --
        dataframe['ema_cross'] = qtpylib.crossed_above(
            dataframe['ema_fast'], dataframe['ema_slow']
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            # Quality gate: signal score >= MIN_SIGNAL_SCORE
            dataframe['signal_score'] >= MIN_SIGNAL_SCORE,
            # Positive expected value
            dataframe['ev'] > 0,
            # Green candle
            dataframe['close'] > dataframe['open'],
            # Not in extreme overbought
            dataframe['rsi'] < 80,
            # Enough volume to get a clean fill
            dataframe['volume'] > 0,
            # ADX shows some trend strength (>15 = any trend, >25 = strong)
            dataframe['adx'] > 15,
            # NaN guards -- never enter on incomplete indicator data
            dataframe['rsi'].notna(),
            dataframe['adx'].notna(),
            dataframe['atr'].notna(),
            dataframe['macdhist'].notna(),
        ]

        dataframe.loc[reduce(lambda a, b: a & b, conditions), 'enter_long'] = 1

        # Store real indicator values in enter_tag for ML training
        def _nan(v, default):
            return default if v != v else float(v)
        def _make_tag(row):
            feat = {
                'sc':   int(row['signal_score']),
                'rsi':  round(float(row['rsi']), 2),
                'rsi1': round(_nan(row['rsi_lag1'], 50.0), 2),
                'mch':  round(float(row['macdhist']),   6),
                'mcs':  round(float(row['macd_slope']), 6),
                'es':   round(float(row['ema_spread']), 6),
                'an':   round(float(row['atr_pct']),    6),
                'volr': round(float(row['volume_ratio']), 3),
                'bbp':  round(float(row['bb_pct']), 4),
                'hod':  float(row.name.hour) if hasattr(row.name, 'hour') else 0.0,
                'dow':  float(row.name.weekday()) if hasattr(row.name, 'weekday') else 0.0,
                'r1':   round(_nan(row['ret_1c'], 0.0), 6),
                'r3':   round(_nan(row['ret_3c'], 0.0), 6),
                'r6':   round(_nan(row['ret_6c'], 0.0), 6),
            }
            tag = json.dumps(feat, separators=(',', ':'))
            return tag[:255]
        for idx in dataframe.index[dataframe['enter_long'] == 1]:
            try:
                dataframe.at[idx, 'enter_tag'] = _make_tag(dataframe.loc[idx])
            except (KeyError, ValueError, TypeError) as _e:
                score_val = int(dataframe.at[idx, 'signal_score'])
                dataframe.at[idx, 'enter_tag'] = f'score_{score_val}'
                logger.warning(f'[PhantomStrategy] _make_tag failed: {_e}')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # EMA crossed down (trend reversal)
                (dataframe['ema_fast'] < dataframe['ema_slow']) &
                # RSI overbought
                (dataframe['rsi'] > 75)
            ) |
            (
                # MACD turned negative
                (dataframe['macdhist'] < 0) &
                (dataframe['macd_slope'] < 0) &
                (dataframe['rsi'] > 60)
            ),
            'exit_long'
        ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """ATR-based stop using ENTRY ATR (from enter_tag).

        NOTE: use_custom_stoploss is currently False, so Freqtrade will not call this method.
        Kept for future activation when switching from trailing stop to ATR-based stop management.
        """
        # Primary: use ATR recorded at trade entry (enter_tag JSON)
        try:
            feat = json.loads(trade.enter_tag or '{}')
            entry_atr_pct = float(feat.get('an', feat.get('atr_norm', 0)))
            if entry_atr_pct > 0.0005:  # sanity check: at least 0.05% ATR
                stop_dist = self.buy_atr_stop.value * entry_atr_pct
                tp1_pct   = ATR_TP1_MULT * entry_atr_pct
                # Tighten to 0.5x stop once TP1 is reached (protect gains)
                if current_profit > tp1_pct:
                    return max(-stop_dist * 0.5, -0.005)
                return -stop_dist
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        # Fallback: current ATR from dataframe (for old trades without real enter_tag)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return self.stoploss
        last_atr_pct = dataframe['atr_pct'].iloc[-1]
        stop_dist = self.buy_atr_stop.value * last_atr_pct
        return -min(stop_dist, 0.03)  # cap fallback at -3%

    def custom_exit(self, pair: str, trade, current_time, current_rate: float,
                    current_profit: float, **kwargs):
        """Take-profit at ATR TP1 -- exits when profit reaches 3x entry ATR."""
        try:
            feat = json.loads(trade.enter_tag or '{}')
            entry_atr_pct = float(feat.get('an', feat.get('atr_norm', 0)))
            if entry_atr_pct > 0.0005:
                tp1_pct = ATR_TP1_MULT * entry_atr_pct
                if current_profit >= tp1_pct:
                    logger.info(f'[PhantomStrategy] {pair} TP1 hit at {current_profit:.4f} (target={tp1_pct:.4f})')
                    return 'tp1_atr'
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
        return None


    # Circuit breaker -- set at startup from config or default
    _initial_wallet: float = 0.0
    _circuit_broken: bool  = False

    def bot_loop_start(self, current_time, **kwargs):
        """Dynamic position sizing + circuit breaker + win rate caching.

        Caches win rate from recent trades for EV calculation (avoids per-candle DB queries).
        Stops trading if drawdown exceeds 20% of initial wallet.
        """
        if Trade is None:
            return

        try:
            # -- Cache win rate from recent closed trades (used in populate_indicators EV calc) --
            closed = Trade.get_trades_proxy(is_open=False)
            recent = sorted(closed, key=lambda t: t.close_date or t.open_date)[-20:]
            if len(recent) >= 10:
                wins = sum(1 for t in recent if (t.close_profit or 0) > 0)
                win_rate = wins / len(recent)
                # Clamp: don't let a lucky streak make us overconfident
                self._cached_win_rate = max(0.40, min(0.70, win_rate))
            else:
                self._cached_win_rate = WIN_RATE_ESTIMATE

            open_value   = sum(t.open_trade_value for t in Trade.get_open_trades())
            available    = self.wallets.get_available_stake_amount()
            total_wallet = available + open_value

            # Record initial wallet on first run
            if self._initial_wallet == 0.0 and total_wallet > 0:
                self._initial_wallet = total_wallet
                logger.info(f'[PhantomStrategy] Initial wallet recorded: {total_wallet:.2f} USDT')

            # -- Circuit breaker: halt if drawdown > 20% from starting wallet --
            MAX_DRAWDOWN = 0.20  # 20% max drawdown before halting
            if self._initial_wallet > 0:
                drawdown = (self._initial_wallet - total_wallet) / self._initial_wallet
                if drawdown >= MAX_DRAWDOWN and not self._circuit_broken:
                    self._circuit_broken = True
                    logger.critical(
                        f'[PhantomStrategy] CIRCUIT BREAKER TRIGGERED -- '
                        f'wallet={total_wallet:.2f} USDT, drawdown={drawdown*100:.1f}% '
                        f'(limit={MAX_DRAWDOWN*100:.0f}%). Setting max_open_trades=0.'
                    )
                    # NOTE: Modifying self.config['max_open_trades'] at runtime is fragile --
                    # Freqtrade may not always respect it. This is a best-effort safeguard.
                    self.config['max_open_trades'] = 0
                    return
                elif drawdown < MAX_DRAWDOWN * 0.5 and self._circuit_broken:
                    # Auto-reset if wallet recovers to 90% of initial (manual review implied)
                    self._circuit_broken = False
                    logger.warning(f'[PhantomStrategy] Circuit breaker reset -- wallet recovered to {total_wallet:.2f} USDT')

            if self._circuit_broken:
                self.config['max_open_trades'] = 0
                return

            dynamic_max = max(1, min(int(total_wallet / MIN_STAKE_PER_TRADE), MAX_OPEN_TRADES_CAP))

            current_max = self.config.get('max_open_trades', -1)
            if current_max != dynamic_max:
                logger.info(
                    f'[PhantomStrategy] max_open_trades: {current_max} -> {dynamic_max} '
                    f'(wallet={total_wallet:.2f} USDT)'
                )
                self.config['max_open_trades'] = dynamic_max
        except (ImportError, AttributeError, ZeroDivisionError) as e:
            logger.debug(f'[PhantomStrategy] bot_loop_start error: {e}')

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                             rate: float, time_in_force: str, current_time,
                             entry_tag=None, side: str = 'long', **kwargs) -> bool:
        """Final gate: signal score, EV, min stake, NaN, price sanity checks."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return False

        last = dataframe.iloc[-1]

        # -- Sanity: reject NaN indicators --
        for col in ('rsi', 'adx', 'signal_score', 'ev', 'atr'):
            val = last.get(col, None)
            if val is None or val != val:  # NaN check
                logger.info(f'[PhantomStrategy] {pair} BLOCKED -- {col} is NaN')
                return False

        # -- Sanity: reject extreme price moves (likely bad data) --
        if len(dataframe) >= 2:
            prev_close = dataframe['close'].iloc[-2]
            if prev_close > 0 and abs(rate / prev_close - 1) > 0.20:
                logger.warning(f'[PhantomStrategy] {pair} BLOCKED -- 20%+ price jump, possible bad data')
                return False

        # -- Min stake check (Binance rejects < 10 USDT) --
        stake_value = amount * rate
        if stake_value < MIN_STAKE_PER_TRADE * 0.90:  # 10% tolerance for float rounding
            logger.info(f'[PhantomStrategy] {pair} BLOCKED -- stake {stake_value:.2f} < {MIN_STAKE_PER_TRADE} USDT minimum')
            return False

        score = last.get('signal_score', 0)
        ev    = last.get('ev', -1)

        if score < MIN_SIGNAL_SCORE:
            logger.info(f'[PhantomStrategy] {pair} BLOCKED -- score {score} < {MIN_SIGNAL_SCORE}')
            return False

        if ev <= 0:
            logger.info(f'[PhantomStrategy] {pair} BLOCKED -- EV {ev:.4f} <= 0')
            return False

        logger.info(f'[PhantomStrategy] {pair} APPROVED -- score {score}/6, EV {ev:.4f}, stake={stake_value:.2f} USDT')
        return True
