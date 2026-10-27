# file: daily_strategy_check.py
"""Daily signal check for the Time-series Momentum + Volatility Scaling strategy.

Run every day at 10:00.
The script:
1. Loads daily BTC data
2. Computes the 30-day return and volatility
3. Determines the signal (LONG/SHORT/OUT)
4. Computes the position size
5. Saves the signal to JSON
6. Analyses the history: how accurate the predictions were
7. Sends the result to Telegram

History is stored in JSON: data/strategy/strategy_history.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Strategy settings
LOOKBACK_DAYS = 30
THRESHOLD = 0.02
MAX_VOL = 0.15

# Files
BASE_DIR = Path(os.environ.get("APP_DIR", Path(__file__).resolve().parent))
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data" / "strategy"))
HISTORY_JSON = DATA_DIR / "strategy_history.json"
PRICE_FILE = DATA_DIR / "btc_prices.csv"

# Create directories on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def ensure_directories():
    """Create the required directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def send_telegram(message: str) -> bool:
    """Send a message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram is not configured.")
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url, 
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, 
            timeout=10
        )
        if resp.status_code == 200:
            print("✅ Message sent to Telegram")
            return True
        else:
            print(f"❌ Send error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {str(e).replace(TELEGRAM_BOT_TOKEN, '<token hidden>') if TELEGRAM_BOT_TOKEN else e}")
        return False


def load_or_download_prices() -> pd.Series:
    """Load BTC prices, or download them if missing."""
    ensure_directories()
    if PRICE_FILE.exists():
        try:
            df = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True, encoding='utf-8')
        except UnicodeDecodeError:
            # Try a different encoding
            df = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True, encoding='latin1')
        print(f"📁 Data loaded: {len(df)} days")
    else:
        print("📥 Initial BTC data download...")
        try:
            import yfinance as yf
            btc = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
            if isinstance(btc.columns, pd.MultiIndex):
                btc.columns = btc.columns.get_level_values(0)
            df = btc[["Close"]].rename(columns={"Close": "close"})
            df.to_csv(PRICE_FILE, encoding='utf-8')
            print(f"✅ Loaded {len(df)} days of data")
        except Exception as e:
            print(f"❌ Download error: {e}")
            raise SystemExit("No access to BTC data")
    return df["close"].dropna()


def update_prices(prices: pd.Series) -> pd.Series:
    """Update prices by appending today's."""
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", period="5d", interval="1d", progress=False)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)
        if len(btc) > 0:
            latest = btc["Close"].iloc[-1]
            latest_date = btc.index[-1]
            if latest_date not in prices.index:
                prices.loc[latest_date] = float(latest)
                prices = prices.sort_index()
                prices.to_csv(PRICE_FILE, encoding='utf-8')
                print(f"✅ Price added: {latest_date.date()} = ${latest:,.0f}")
    except Exception as e:
        print(f"⚠️ Could not update: {e}")
    return prices


def calculate_signal(prices: pd.Series) -> Optional[Dict]:
    """Compute the strategy signal."""
    if len(prices) < LOOKBACK_DAYS + 5:
        print(f"❌ Not enough data: {len(prices)} days")
        return None

    current_price = prices.iloc[-1]
    current_date = prices.index[-1]
    price_lookback = prices.iloc[-LOOKBACK_DAYS]
    past_return = (current_price - price_lookback) / price_lookback

    daily_returns = prices.pct_change().dropna()
    returns_last_30 = daily_returns.iloc[-LOOKBACK_DAYS:]
    volatility = returns_last_30.std() * np.sqrt(365)

    if past_return > THRESHOLD:
        signal = 1
        action = "LONG"
    elif past_return < -THRESHOLD:
        signal = -1
        action = "SHORT"
    else:
        signal = 0
        action = "OUT"

    position = signal * min(1.0, MAX_VOL / volatility) if signal != 0 else 0.0

    return {
        "date": str(current_date.date()),
        "price": float(current_price),
        "past_return": float(past_return),
        "volatility": float(volatility),
        "signal": signal,
        "action": action,
        "position": float(position),
        "position_pct": float(position * 100),
        "timestamp": datetime.now().isoformat(),
    }


def load_history() -> List[Dict]:
    """Load the history from JSON with the right encoding."""
    if HISTORY_JSON.exists():
        try:
            # Try UTF-8 with BOM
            with open(HISTORY_JSON, 'r', encoding='utf-8-sig') as fh:
                return json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Try plain UTF-8
                with open(HISTORY_JSON, 'r', encoding='utf-8') as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    # Try cp1251 (Windows)
                    with open(HISTORY_JSON, 'r', encoding='cp1251') as fh:
                        return json.load(fh)
                except:
                    print("⚠️ Could not read the JSON, creating a new one")
                    return []
    return []


def save_history(history: List[Dict]) -> None:
    """Save the history to JSON with the right encoding."""
    # Use UTF-8 with BOM for Windows compatibility
    with open(HISTORY_JSON, 'w', encoding='utf-8-sig') as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)


def calculate_returns(signal_price: float, future_prices: pd.Series, days: List[int]) -> Dict[str, float]:
    """Compute returns for different horizons."""
    returns = {}
    for d in days:
        if len(future_prices) >= d:
            future_price = future_prices.iloc[d - 1]
            returns[f"{d}d_return"] = (future_price - signal_price) / signal_price
    return returns


def check_prediction(action: str, returns: Dict[str, float]) -> Dict[str, str]:
    """Check whether a signal was correct."""
    results = {}
    for period, ret in returns.items():
        days = period.split("d")[0]
        if action == "LONG":
            results[f"{days}d"] = "OK" if ret > 0 else "BAD"
        elif action == "SHORT":
            results[f"{days}d"] = "OK" if ret < 0 else "BAD"
        else:  # OUT
            results[f"{days}d"] = "SKIP"
    return results


def analyze_history(history: List[Dict], current_prices: pd.Series) -> Dict:
    """Analyse the signal history."""
    if not history:
        return {
            "total_signals": 0,
            "analyzed": 0,
            "pending": 0,
            "correct_7d": 0,
            "wrong_7d": 0,
            "accuracy_7d": None,
            "avg_return_long": None,
            "avg_return_short": None,
            "last_signals": [],
        }

    analyzed = 0
    correct_7d = 0
    wrong_7d = 0
    pending = 0
    long_returns = []
    short_returns = []
    last_signals = []

    for entry in history:
        signal_date = pd.Timestamp(entry["date"])
        action = entry["action"]
        signal_price = entry["price"]

        # Get future prices
        future_prices = current_prices[current_prices.index > signal_date]

        if len(future_prices) == 0:
            pending += 1
            entry["status"] = "pending"
            continue

        # Compute returns
        returns = calculate_returns(signal_price, future_prices, [1, 3, 7, 14, 30])
        
        # Check the prediction (ASCII instead of emoji)
        results = check_prediction(action, returns)
        entry["results"] = results
        entry["last_checked_date"] = str(current_prices.index[-1].date())
        entry["status"] = "analyzed"

        # Collect statistics for the 7-day horizon
        if "7d" in results:
            if results["7d"] == "OK":
                correct_7d += 1
            elif results["7d"] == "BAD":
                wrong_7d += 1

        # Collect returns for statistics
        if action == "LONG" and "7d_return" in returns:
            long_returns.append(returns["7d_return"])
        elif action == "SHORT" and "7d_return" in returns:
            short_returns.append(-returns["7d_return"])

        analyzed += 1
        last_signals.append({
            "date": entry["date"],
            "action": action,
            "price": signal_price,
            "result_7d": results.get("7d", "PENDING"),
        })

    # Compute accuracy
    accuracy_7d = round(correct_7d / (correct_7d + wrong_7d) * 100, 1) if (correct_7d + wrong_7d) > 0 else None
    avg_return_long = np.mean(long_returns) if long_returns else None
    avg_return_short = np.mean(short_returns) if short_returns else None

    return {
        "total_signals": len(history),
        "analyzed": analyzed,
        "pending": pending,
        "correct_7d": correct_7d,
        "wrong_7d": wrong_7d,
        "accuracy_7d": accuracy_7d,
        "avg_return_long": avg_return_long,
        "avg_return_short": avg_return_short,
        "last_signals": last_signals[-5:],
    }


def format_position_desc(position_pct: float) -> str:
    """Position size description."""
    if position_pct > 90:
        return "FULL"
    elif position_pct > 50:
        return "MEDIUM"
    elif position_pct > 25:
        return "SMALL"
    elif position_pct > 0:
        return "MINIMAL"
    return "NONE"


def format_console(signal: Dict, analysis: Dict) -> str:
    """Format the console message."""
    lines = []
    lines.append("=" * 70)
    lines.append("DAILY STRATEGY CHECK")
    lines.append("=" * 70)
    lines.append(f"Date: {signal['date']}")
    lines.append(f"BTC price: ${signal['price']:,.0f}")
    lines.append(f"30-day return: {signal['past_return']:+.2%}")
    lines.append(f"Volatility: {signal['volatility']:.2%}")
    lines.append("=" * 70)

    if signal["action"] == "LONG":
        lines.append("SIGNAL: BUY (LONG)")
    elif signal["action"] == "SHORT":
        lines.append("SIGNAL: SELL (SHORT)")
    else:
        lines.append("SIGNAL: NO POSITION (OUT)")

    position_desc = format_position_desc(signal['position_pct'])
    lines.append(f"Position size: {signal['position_pct']:.1f}% ({position_desc})")
    lines.append("=" * 70)

    # Analysis
    lines.append("\nHISTORY ANALYSIS:")
    lines.append(f"Total signals: {analysis['total_signals']}")
    lines.append(f"Analysed: {analysis['analyzed']}")
    lines.append(f"Awaiting check: {analysis['pending']}")
    
    if analysis.get("accuracy_7d") is not None:
        lines.append(f"\nAccuracy (7 days): {analysis['accuracy_7d']}%")
        lines.append(f"Correct: {analysis['correct_7d']} | Wrong: {analysis['wrong_7d']}")
    
    if analysis.get("avg_return_long") is not None:
        lines.append(f"\nMean LONG return (7d): {analysis['avg_return_long']:+.2%}")
    if analysis.get("avg_return_short") is not None:
        lines.append(f"Mean SHORT return (7d): {analysis['avg_return_short']:+.2%}")

    # Latest signals
    if analysis.get("last_signals"):
        lines.append("\nLatest signals:")
        for sig in analysis["last_signals"]:
            lines.append(f"  {sig['date']} | {sig['action']:5s} | ${sig['price']:,.0f} | 7d: {sig['result_7d']}")

    return "\n".join(lines)


def format_telegram(signal: Dict, analysis: Dict) -> str:
    """Format the Telegram message."""
    emoji = {"LONG": "🟢", "SHORT": "🔴", "OUT": "⚪"}.get(signal["action"], "❓")
    action_text = {
        "LONG": "BUY (LONG)",
        "SHORT": "SELL (SHORT)",
        "OUT": "NO POSITION (OUT)"
    }[signal["action"]]
    
    position_desc = format_position_desc(signal['position_pct'])

    msg = f"""<b>📊 BTC Strategy Signal</b>

<b>{emoji} {action_text}</b>

📅 <b>Date:</b> {signal['date']}
💰 <b>Price:</b> ${signal['price']:,.0f}

📈 <b>30d return:</b> {signal['past_return']:+.2%}
📊 <b>Volatility:</b> {signal['volatility']:.2%}
📐 <b>Position:</b> {signal['position_pct']:.1f}% ({position_desc})

━━━━━━━━━━━━━━━
📋 <b>Statistics:</b>
Total: {analysis['total_signals']} | Checked: {analysis['analyzed']}"""

    if analysis.get("accuracy_7d") is not None:
        msg += f"\n🎯 Accuracy (7d): <b>{analysis['accuracy_7d']}%</b>"
        msg += f"\n✅ {analysis['correct_7d']} | ❌ {analysis['wrong_7d']}"

    if analysis.get("avg_return_long") is not None:
        msg += f"\n📈 Avg LONG return: {analysis['avg_return_long']:+.2%}"
    if analysis.get("avg_return_short") is not None:
        msg += f"\n📉 Avg SHORT return: {analysis['avg_return_short']:+.2%}"

    return msg


def main():
    """Main function."""
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load data
    prices = load_or_download_prices()
    
    # 2. Update prices
    prices = update_prices(prices)

    # 3. Compute the signal
    signal = calculate_signal(prices)
    if signal is None:
        print("Not enough data to compute")
        return 1

    # 4. Load history
    history = load_history()

    # 5. Add the new signal (replace today's record if present)
    history = [h for h in history if h["date"] != signal["date"]]
    history.append(signal)
    save_history(history)
    print(f"Signal saved to history ({len(history)} records)")

    # 6. Analyse history
    analysis = analyze_history(history, prices)

    # 7. Print to screen
    console_msg = format_console(signal, analysis)
    print("\n" + console_msg)

    # 8. Telegram is off by default (the daily BTC message was turned off); the signal journal keeps updating.
    #    To turn messages back on: BTC_TELEGRAM=1 in .env
    if os.environ.get("BTC_TELEGRAM", "0") == "1":
        telegram_msg = format_telegram(signal, analysis)
        send_telegram(telegram_msg)
    else:
        print("Telegram for BTC is off (BTC_TELEGRAM != 1).")

    print(f"\nDone! Next run tomorrow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())