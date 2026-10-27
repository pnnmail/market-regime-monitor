"""
DURATION / VOLATILITY REGIME MONITOR v2
SECOND MARKET THEORY

============================================================
PURPOSE
============================================================

This script analyses not a single indicator but the
RELATIONSHIPS between several large market segments:

    1. Long US Treasury bonds
       TLT / IEF / SHY

    2. Volatility
       VIX

    3. Equities
       SPY

    4. Credit market
       HYG / LQD

    5. Macroeconomic indicators
       10Y Treasury yield / DXY


============================================================
MAIN HYPOTHESIS
============================================================

A particularly interesting situation arises when:

    TLT ↑
    VIX ↓
    SPY ↑
    CREDIT ↑

That is:

    bonds start to strengthen,
    volatility stays very low,
    equities keep rising,
    the credit market shows no stress.

This does NOT automatically mean:

    "the market is about to fall".

It means:

    "different market segments are starting to send
     different macroeconomic messages".

This divergence is the main object
of observation for this theory.


============================================================
WHY THIS IS INTERESTING
============================================================

TLT is especially sensitive to the market's long-term expectations
regarding:

    - future economic growth;
    - inflation;
    - interest rates;
    - Federal Reserve policy;
    - demand for safe-haven assets.

If long Treasuries start rising, the bond market may be
pricing in lower future rates or an economic slowdown.

But if at the same time:

    SPY keeps rising,
    VIX is very low,
    HYG/LQD stays strong,

then the equity market does not yet confirm a serious
macroeconomic risk.


============================================================
IMPORTANT
============================================================

This script is NOT a price forecaster.

It answers a different question:

    "How consistent are the different parts of the market with each other?"

The stronger the divergence between them, the more interesting the situation
is for further observation.

It is especially important to distinguish:

    MACRO DISLOCATION

from

    RISK-OFF.

The first means:

    "the market is starting to diverge in its signals".

The second means:

    "different segments already confirm stress in sync".


============================================================
REGIMES
============================================================

🟢 RISK-ON / HEALTHY

Equities strong.
Credit strong.
VIX shows no serious stress.
Bonds give no strong opposite signal.

This is the most normal bullish configuration.


🟡 DURATION WARNING

Long Treasuries start to strengthen,
but equities remain strong for now.

This is NOT risk-off yet.

It is a warning:

    "the bond market is starting to say
     something different from the equity market".


🟠 MACRO DISLOCATION

The most interesting regime of this theory.

At the same time:

    TLT strong
    SPY strong
    VIX very low
    CREDIT healthy

This means:

    the equity market is calm and strong for now,
    but the long-bond market
    is starting to price in a different macroeconomic picture.

This is NOT an immediate sell signal.

It is a state that must be watched.


🔴 RISK-OFF

At the same time:

    TLT ↑
    SPY ↓
    CREDIT ↓
    VIX ↑

Here the divergence ends.

The defensive market and risk assets are already starting
to move in the same direction.

This is a much more serious signal.


🟣 COMPRESSION / COMPLACENCY

Equities strong.
Credit strong.
VIX very low.

Meanwhile the market prices in practically no stress.

This means:

    "the market is very calm".

If TLT starts strengthening at the same time,
a potentially more interesting
MACRO DISLOCATION.


============================================================
USAGE
============================================================

python theories/duration_volatility.py

Historical date:

python theories/duration_volatility.py --date 2026-08-31

Telegram:

python theories/duration_volatility.py --telegram

Saving:

python theories/duration_volatility.py --save

Both:

python theories/duration_volatility.py --save --telegram
"""

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

TZ = ZoneInfo("America/New_York")

HISTORY_PERIOD = "3y"

MA20 = 20
MA50 = 50
MA200 = 200


# ============================================================
# TICKERS
# ============================================================

TICKERS = {
    "SPY": "SPY",
    "TLT": "TLT",
    "IEF": "IEF",
    "SHY": "SHY",
    "HYG": "HYG",
    "LQD": "LQD",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Duration / Volatility Regime Monitor"
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Historical date YYYY-MM-DD",
    )

    parser.add_argument(
        "--period",
        default=HISTORY_PERIOD,
        help="Yahoo Finance history period",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to CSV",
    )

    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send report to Telegram",
    )

    return parser.parse_args()


# ============================================================
# DOWNLOAD
# ============================================================

def download_data(period):

    print()
    print("Loading market macro data...")
    print()

    tickers = list(TICKERS.values())

    try:

        data = yf.download(
            tickers=tickers,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )

    except Exception as exc:

        raise RuntimeError(
            f"Yahoo Finance download error: {exc}"
        )

    if data.empty:

        raise RuntimeError(
            "Yahoo Finance returned no data."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if "Close" not in data.columns.get_level_values(0):

            raise RuntimeError(
                "Close prices not found."
            )

        close = data["Close"].copy()

    else:

        if "Close" not in data.columns:

            raise RuntimeError(
                "Close prices not found."
            )

        close = data[
            ["Close"]
        ].copy()

        if len(tickers) == 1:

            close.columns = tickers

    reverse = {
        value: key
        for key, value in TICKERS.items()
    }

    close = close.rename(
        columns=reverse
    )

    close.index = pd.to_datetime(
        close.index
    )

    close = close.sort_index()

    print(
        f"Trading days loaded: {len(close)}"
    )

    print(
        f"Period: {close.index[0].date()} → "
        f"{close.index[-1].date()}"
    )

    return close


# ============================================================
# DATE
# ============================================================

def select_date(
    close,
    requested,
):

    dates = close.index

    if requested:

        target = pd.Timestamp(
            requested
        )

        available = dates[
            dates <= target
        ]

        if len(available) == 0:

            raise RuntimeError(
                "No data for the given date."
            )

        return available[-1]

    return dates[-1]


# ============================================================
# HELPERS
# ============================================================

def pct_change(
    series,
    days,
):

    if len(series) <= days:

        return np.nan

    current = series.iloc[-1]
    previous = series.iloc[-days - 1]

    if pd.isna(current) or pd.isna(previous):

        return np.nan

    return (
        current / previous - 1
    ) * 100


def moving_average(
    series,
    window,
):

    return float(
        series.tail(window).mean()
    )


# ============================================================
# DURATION
# ============================================================

def calculate_duration(
    close,
    date,
):

    data = close.loc[
        :date
    ]

    tlt = data["TLT"].dropna()
    ief = data["IEF"].dropna()
    shy = data["SHY"].dropna()

    if len(tlt) < MA200:

        raise RuntimeError(
            "Not enough TLT history."
        )

    tlt_current = float(
        tlt.iloc[-1]
    )

    tlt_ma20 = moving_average(
        tlt,
        MA20,
    )

    tlt_ma50 = moving_average(
        tlt,
        MA50,
    )

    tlt_ma200 = moving_average(
        tlt,
        MA200,
    )

    tlt_20d = pct_change(
        tlt,
        20,
    )

    tlt_63d = pct_change(
        tlt,
        63,
    )

    tlt_252d = pct_change(
        tlt,
        252,
    )

    tlt_ief_ratio = (
        tlt / ief
    ).dropna()

    ratio_current = float(
        tlt_ief_ratio.iloc[-1]
    )

    ratio_ma20 = moving_average(
        tlt_ief_ratio,
        MA20,
    )

    ratio_ma50 = moving_average(
        tlt_ief_ratio,
        MA50,
    )

    ratio_63d = pct_change(
        tlt_ief_ratio,
        63,
    )

    tlt_shy_ratio = (
        tlt / shy
    ).dropna()

    tlt_shy_current = float(
        tlt_shy_ratio.iloc[-1]
    )

    tlt_shy_63d = pct_change(
        tlt_shy_ratio,
        63,
    )

    return {

        "tlt": tlt_current,

        "tlt_ma20":
            tlt_ma20,

        "tlt_ma50":
            tlt_ma50,

        "tlt_ma200":
            tlt_ma200,

        "tlt_20d":
            tlt_20d,

        "tlt_63d":
            tlt_63d,

        "tlt_252d":
            tlt_252d,

        "tlt_above_20":
            tlt_current > tlt_ma20,

        "tlt_above_50":
            tlt_current > tlt_ma50,

        "tlt_above_200":
            tlt_current > tlt_ma200,

        "tlt_ief_ratio":
            ratio_current,

        "tlt_ief_ma20":
            ratio_ma20,

        "tlt_ief_ma50":
            ratio_ma50,

        "tlt_ief_63d":
            ratio_63d,

        "tlt_ief_above_20":
            ratio_current > ratio_ma20,

        "tlt_ief_above_50":
            ratio_current > ratio_ma50,

        "tlt_shy_ratio":
            tlt_shy_current,

        "tlt_shy_63d":
            tlt_shy_63d,
    }


# ============================================================
# VIX
# ============================================================

def calculate_vix(
    close,
    date,
):

    data = close.loc[
        :date,
        "VIX"
    ].dropna()

    if len(data) < MA50:

        raise RuntimeError(
            "Not enough VIX history."
        )

    current = float(
        data.iloc[-1]
    )

    ma20 = moving_average(
        data,
        MA20,
    )

    ma50 = moving_average(
        data,
        MA50,
    )

    percentile_window = data.tail(
        252
    )

    percentile = (
        percentile_window < current
    ).mean() * 100

    change_20d = pct_change(
        data,
        20,
    )

    return {

        "current":
            current,

        "ma20":
            ma20,

        "ma50":
            ma50,

        "percentile":
            float(percentile),

        "change_20d":
            change_20d,

        "below_20":
            current < ma20,

        "below_50":
            current < ma50,
    }


# ============================================================
# EQUITIES
# ============================================================

def calculate_equities(
    close,
    date,
):

    spy = close.loc[
        :date,
        "SPY"
    ].dropna()

    current = float(
        spy.iloc[-1]
    )

    ma20 = moving_average(
        spy,
        MA20,
    )

    ma50 = moving_average(
        spy,
        MA50,
    )

    ma200 = moving_average(
        spy,
        MA200,
    )

    return {

        "spy":
            current,

        "ma20":
            ma20,

        "ma50":
            ma50,

        "ma200":
            ma200,

        "above20":
            current > ma20,

        "above50":
            current > ma50,

        "above200":
            current > ma200,

        "return20":
            pct_change(spy, 20),

        "return63":
            pct_change(spy, 63),

        "return252":
            pct_change(spy, 252),
    }


# ============================================================
# CREDIT
# ============================================================

def calculate_credit(
    close,
    date,
):

    data = close.loc[
        :date
    ]

    hyg = data["HYG"].dropna()
    lqd = data["LQD"].dropna()

    if len(hyg) < MA50:

        raise RuntimeError(
            "Not enough credit market history."
        )

    hyg_lqd = (
        hyg / lqd
    ).dropna()

    current = float(
        hyg_lqd.iloc[-1]
    )

    ma20 = moving_average(
        hyg_lqd,
        MA20,
    )

    ma50 = moving_average(
        hyg_lqd,
        MA50,
    )

    return {

        "hyg":
            float(hyg.iloc[-1]),

        "lqd":
            float(lqd.iloc[-1]),

        "hyg_lqd":
            current,

        "hyg_lqd_ma20":
            ma20,

        "hyg_lqd_ma50":
            ma50,

        "hyg_lqd_above20":
            current > ma20,

        "hyg_lqd_above50":
            current > ma50,

        "hyg_lqd_20d":
            pct_change(
                hyg_lqd,
                20,
            ),

        "hyg_lqd_63d":
            pct_change(
                hyg_lqd,
                63,
            ),
    }


# ============================================================
# MACRO
# ============================================================

def calculate_macro(
    close,
    date,
):

    tnx = close.loc[
        :date,
        "TNX"
    ].dropna()

    dxy = close.loc[
        :date,
        "DXY"
    ].dropna()

    result = {}

    if len(tnx) > 20:

        result["tnx"] = float(
            tnx.iloc[-1]
        )

        result["tnx_20d"] = pct_change(
            tnx,
            20,
        )

    else:

        result["tnx"] = np.nan
        result["tnx_20d"] = np.nan

    if len(dxy) > 20:

        result["dxy"] = float(
            dxy.iloc[-1]
        )

        result["dxy_20d"] = pct_change(
            dxy,
            20,
        )

    else:

        result["dxy"] = np.nan
        result["dxy_20d"] = np.nan

    return result


# ============================================================
# REGIME
# ============================================================

def determine_regime(
    duration,
    vix,
    equities,
    credit,
):

    duration_strong = (
        duration["tlt_above_50"]
        and duration["tlt_63d"] > 0
    )

    duration_relative = (
        duration["tlt_ief_above_20"]
        and duration["tlt_shy_63d"] > 0
    )

    vix_complacent = (
        vix["current"] < 15
    )

    vix_low = (
        vix["current"] < 20
    )

    equity_strong = (
        equities["above50"]
        and equities["above200"]
    )

    equity_weak = (
        not equities["above50"]
    )

    credit_healthy = (
        credit["hyg_lqd_above20"]
    )

    credit_weak = (
        not credit["hyg_lqd_above50"]
    )

    duration_score = sum(
        [
            duration_strong,
            duration_relative,
        ]
    )

    equity_score = sum(
        [
            equities["above20"],
            equities["above50"],
            equities["above200"],
        ]
    )

    credit_score = sum(
        [
            credit["hyg_lqd_above20"],
            credit["hyg_lqd_above50"],
        ]
    )

    # ========================================================
    # MACRO DISLOCATION
    # ========================================================

    if (
        duration_score >= 1
        and vix_complacent
        and equity_strong
        and credit_healthy
    ):

        regime = "🟠 MACRO DISLOCATION"

        explanation = (
            "The main divergence this theory looks for is "
            "happening now. Long Treasuries "
            "are strengthening, while equities, credit "
            "and volatility keep a calm "
            "configuration. This means the bond market "
            "may be pricing in weaker future growth, "
            "lower inflation or future rate cuts, "
            "while the equity market does not yet confirm "
            "the need for a defensive regime. This does not mean "
            "an immediate fall in equities. It means "
            "one needs to watch which of the markets "
            "turns out to be right."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            explanation,
        )

    # ========================================================
    # RISK-OFF
    # ========================================================

    if (
        duration_score >= 1
        and equity_weak
        and credit_weak
        and not vix_complacent
    ):

        regime = "🔴 RISK-OFF"

        explanation = (
            "Here the divergence has already turned into a synchronised "
            "defensive regime. Long bonds are strengthening, "
            "equities are weakening, credit is deteriorating, "
            "and VIX is no longer low. This is a substantially "
            "more serious signal than an ordinary strengthening "
            "of TLT. Several independent market segments "
            "simultaneously confirm rising risk."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            explanation,
        )

    # ========================================================
    # DURATION WARNING
    # ========================================================

    if (
        duration_score >= 1
        and equity_strong
    ):

        regime = "🟡 DURATION WARNING"

        explanation = (
            "Long Treasuries show strength, but the equity "
            "market remains strong for now. This is an early "
            "macroeconomic warning, not "
            "a reversal signal. The bond market is starting "
            "to send a message that differs from "
            "the equity market's message. If credit and VIX "
            "start to deteriorate at the same time, the warning "
            "may turn into MACRO DISLOCATION or RISK-OFF."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            explanation,
        )

    # ========================================================
    # COMPRESSION
    # ========================================================

    if (
        vix_complacent
        and equity_strong
        and credit_healthy
    ):

        regime = "🟣 COMPRESSION / COMPLACENCY"

        explanation = (
            "The market is in a very calm configuration: "
            "equities strong, credit healthy and VIX "
            "very low. This means a low current "
            "assessment of risk by the market. By itself "
            "this is not a bearish signal. However, "
            "if long Treasuries start strengthening sharply against "
            "this background, an important divergence will arise "
            "between the bond market and the risk market."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            explanation,
        )

    # ========================================================
    # HEALTHY RISK-ON
    # ========================================================

    if (
        equity_strong
        and credit_healthy
    ):

        regime = "🟢 RISK-ON / HEALTHY"

        explanation = (
            "Equities and credit move consistently "
            "and show no serious stress. "
            "Long Treasuries do not create a strong enough "
            "opposite signal to form "
            "a macroeconomic divergence."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            explanation,
        )

    return (
        "🟡 MIXED",
        duration_score,
        equity_score,
        credit_score,
        "Signals from different market segments are mixed. "
        "Neither full risk-on nor full risk-off "
        "is confirmed yet.",
    )


# ============================================================
# INTERPRETATION ENGINE
# ============================================================

def build_interpretation(
    duration,
    vix,
    equities,
    credit,
    regime,
):

    lines = []

    lines.append(
        "8️⃣ WHAT IS HAPPENING NOW"
    )

    lines.append(
        "──────────────────────────────"
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if duration["tlt_above_50"]:

        lines.append(
            "• BONDS: long Treasuries are "
            "above the 50DMA and show positive "
            "63-day momentum. This means that "
            "demand for duration cannot currently be considered "
            "a random one-day move."
        )

        if duration["tlt_ief_above_20"]:

            lines.append(
                "  → TLT is also stronger relative to IEF. "
                "I.e. within Treasuries demand is shifted "
                "towards longer duration."
            )

    else:

        lines.append(
            "• BONDS: no strong confirmation "
            "of long-duration strength yet."
        )

    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    if vix["current"] < 15:

        lines.append(
            "• VIX: very low. The options market "
            "does not expect high short-term volatility."
        )

        lines.append(
            "  → Important: a low VIX does NOT mean the "
            "market must keep rising. "
            "It only means that the current price of protection "
            "against a sharp move is low."
        )

    elif vix["current"] < 20:

        lines.append(
            "• VIX: low/moderate. No panic "
            "demand for protection yet."
        )

    elif vix["current"] < 30:

        lines.append(
            "• VIX: elevated. The market is starting "
            "to price risk more substantially."
        )

    else:

        lines.append(
            "• VIX: high. Volatility confirms "
            "serious market stress."
        )

    # --------------------------------------------------------
    # Equities
    # --------------------------------------------------------

    if equities["above50"] and equities["above200"]:

        lines.append(
            "• EQUITIES: SPY is above the 50DMA and 200DMA. "
            "The medium- and long-term trends remain "
            "positive."
        )

    elif equities["above200"]:

        lines.append(
            "• EQUITIES: the long-term trend is still positive, "
            "but the medium-term structure is weakening."
        )

    else:

        lines.append(
            "• EQUITIES: SPY does not confirm a strong "
            "long-term structure."
        )

    # --------------------------------------------------------
    # Credit
    # --------------------------------------------------------

    if credit["hyg_lqd_above20"]:

        lines.append(
            "• CREDIT: HYG/LQD above the 20DMA. "
            "The credit market does not yet show "
            "a strong flight of investors from riskier "
            "bonds."
        )

    else:

        lines.append(
            "• CREDIT: HYG/LQD below the 20DMA. "
            "Caution is appearing in the credit market."
        )

    # --------------------------------------------------------
    # Final logic
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "9️⃣ PUTTING IT INTO ONE PICTURE"
    )

    lines.append(
        "──────────────────────────────"
    )

    if regime == "🟠 MACRO DISLOCATION":

        lines.append(
            "The market is now in a state of "
            "MACROECONOMIC DIVERGENCE."
        )

        lines.append(
            ""
        )

        lines.append(
            "Main observation:"
        )

        lines.append(
            "TLT points to a potentially weaker "
            "future macro backdrop, "
            "while SPY + CREDIT + VIX say for now "
            "that the current market remains calm."
        )

        lines.append(
            ""
        )

        lines.append(
            "What this means:"
        )

        lines.append(
            "→ This is NOT a confirmed risk-off."
        )

        lines.append(
            "→ This is NOT proof of a future decline."
        )

        lines.append(
            "→ This is a divergence that needs watching."
        )

        lines.append(
            ""
        )

        lines.append(
            "Key question:"
        )

        lines.append(
            "Who will confirm the signal first — "
            "will equities/credit weaken, or will TLT stop "
            "strengthening?"
        )

    elif regime == "🟡 DURATION WARNING":

        lines.append(
            "The market remains predominantly "
            "risk-on, but Treasuries are starting to give "
            "an opposite macro signal."
        )

        lines.append(
            "Watch VIX and HYG/LQD first of all."
        )

        lines.append(
            "If credit deteriorates and VIX starts rising, "
            "the warning will become much more serious."
        )

    elif regime == "🔴 RISK-OFF":

        lines.append(
            "Several independent market segments "
            "simultaneously confirm a transition "
            "into a defensive regime."
        )

        lines.append(
            "This is no longer just a divergence between "
            "bonds and equities."
        )

        lines.append(
            "Treasuries, VIX and the credit market "
            "confirm rising risk."
        )

    elif regime == "🟣 COMPRESSION / COMPLACENCY":

        lines.append(
            "The market is in a regime of high "
            "complacency: equities and credit are strong, "
            "and VIX is very low."
        )

        lines.append(
            "This is not a bearish signal yet."
        )

        lines.append(
            "But a strong upward TLT move "
            "against this background would be especially interesting."
        )

    elif regime == "🟢 RISK-ON / HEALTHY":

        lines.append(
            "Most key market segments "
            "confirm normal risk-on."
        )

        lines.append(
            "Equities and credit are consistent, and there is no strong "
            "contradictory signal from "
            "Treasuries yet."
        )

    else:

        lines.append(
            "The market is in a transition zone."
        )

        lines.append(
            "Signals do not add up to a stable "
            "risk-on or risk-off regime."
        )

    return lines


# ============================================================
# REPORT
# ============================================================

def build_report(
    date,
    duration,
    vix,
    equities,
    credit,
    macro,
    regime,
    duration_score,
    equity_score,
    credit_score,
    explanation,
):

    lines = []

    lines.append(
        "📊 DURATION / VOLATILITY REGIME MONITOR"
    )

    lines.append(
        "SECOND MARKET THEORY"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"Market date: {date.date()}"
    )

    lines.append("")

    # ========================================================
    # DURATION
    # ========================================================

    lines.append(
        "1️⃣ LONG US TREASURY BONDS"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"TLT: {duration['tlt']:.2f}"
    )

    lines.append(
        f"20DMA: {duration['tlt_ma20']:.2f} "
        f"({'✓ above' if duration['tlt_above_20'] else '✗ below'})"
    )

    lines.append(
        f"50DMA: {duration['tlt_ma50']:.2f} "
        f"({'✓ above' if duration['tlt_above_50'] else '✗ below'})"
    )

    lines.append(
        f"200DMA: {duration['tlt_ma200']:.2f} "
        f"({'✓ above' if duration['tlt_above_200'] else '✗ below'})"
    )

    lines.append(
        f"TLT 20-day return: "
        f"{duration['tlt_20d']:+.2f}%"
    )

    lines.append(
        f"TLT 63-day return: "
        f"{duration['tlt_63d']:+.2f}%"
    )

    lines.append(
        f"TLT 252-day return: "
        f"{duration['tlt_252d']:+.2f}%"
    )

    lines.append("")

    lines.append(
        "What this means:"
    )

    if duration["tlt_above_50"]:

        lines.append(
            "→ Long-duration Treasuries currently have "
            "a positive medium-term structure."
        )

    else:

        lines.append(
            "→ Long-duration Treasuries do not yet have "
            "stable medium-term strength."
        )

    if duration["tlt_63d"] > 0:

        lines.append(
            "→ Over the last ~3 months TLT has risen."
        )

    else:

        lines.append(
            "→ Over the last ~3 months TLT has fallen."
        )

    lines.append("")

    lines.append(
        "TLT / IEF — long duration vs "
        "intermediate duration:"
    )

    lines.append(
        f"Ratio: {duration['tlt_ief_ratio']:.4f}"
    )

    lines.append(
        f"20DMA: {duration['tlt_ief_ma20']:.4f} "
        f"({'✓' if duration['tlt_ief_above_20'] else '✗'})"
    )

    lines.append(
        f"50DMA: {duration['tlt_ief_ma50']:.4f} "
        f"({'✓' if duration['tlt_ief_above_50'] else '✗'})"
    )

    lines.append(
        f"63-day change: "
        f"{duration['tlt_ief_63d']:+.2f}%"
    )

    if duration["tlt_ief_above_20"]:

        lines.append(
            "→ Long duration is currently stronger than "
            "intermediate relative to its recent average."
        )

    else:

        lines.append(
            "→ No clear relative advantage "
            "for long duration yet."
        )

    lines.append("")

    # ========================================================
    # VIX
    # ========================================================

    lines.append(
        "2️⃣ VIX — EXPECTED VOLATILITY"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"VIX: {vix['current']:.2f}"
    )

    lines.append(
        f"20DMA: {vix['ma20']:.2f}"
    )

    lines.append(
        f"50DMA: {vix['ma50']:.2f}"
    )

    lines.append(
        f"Position within 252 days: "
        f"{vix['percentile']:.1f} percentile"
    )

    lines.append(
        f"VIX 20-day change: "
        f"{vix['change_20d']:+.2f}%"
    )

    if vix["current"] < 15:

        vix_regime = "🟢 VERY LOW"

        vix_meaning = (
            "The market prices in practically no "
            "strong short-term volatility."
        )

    elif vix["current"] < 20:

        vix_regime = "🟢 LOW"

        vix_meaning = (
            "The market shows no pronounced "
            "fear."
        )

    elif vix["current"] < 30:

        vix_regime = "🟡 ELEVATED"

        vix_meaning = (
            "Demand for protection and volatility expectations "
            "have risen noticeably."
        )

    else:

        vix_regime = "🔴 HIGH"

        vix_meaning = (
            "The market is in a pronounced "
            "stress regime."
        )

    lines.append(
        f"VIX regime: {vix_regime}"
    )

    lines.append(
        f"What this means: {vix_meaning}"
    )

    lines.append("")

    # ========================================================
    # EQUITIES
    # ========================================================

    lines.append(
        "3️⃣ EQUITY MARKET — SPY"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"SPY: {equities['spy']:.2f}"
    )

    lines.append(
        f"20DMA: {equities['ma20']:.2f} "
        f"({'✓ above' if equities['above20'] else '✗ below'})"
    )

    lines.append(
        f"50DMA: {equities['ma50']:.2f} "
        f"({'✓ above' if equities['above50'] else '✗ below'})"
    )

    lines.append(
        f"200DMA: {equities['ma200']:.2f} "
        f"({'✓ above' if equities['above200'] else '✗ below'})"
    )

    lines.append(
        f"20D: {equities['return20']:+.2f}%"
    )

    lines.append(
        f"63D: {equities['return63']:+.2f}%"
    )

    lines.append(
        f"252D: {equities['return252']:+.2f}%"
    )

    lines.append("")

    if equities["above50"] and equities["above200"]:

        lines.append(
            "→ SPY keeps a strong medium-term "
            "and long-term structure."
        )

    elif equities["above200"]:

        lines.append(
            "→ The long-term trend is still positive, "
            "but the medium-term structure is deteriorating."
        )

    else:

        lines.append(
            "→ SPY does not confirm a strong "
            "long-term trend."
        )

    # ========================================================
    # CREDIT
    # ========================================================

    lines.append("")

    lines.append(
        "4️⃣ CREDIT MARKET"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"HYG: {credit['hyg']:.2f}"
    )

    lines.append(
        f"LQD: {credit['lqd']:.2f}"
    )

    lines.append(
        f"HYG/LQD: {credit['hyg_lqd']:.4f}"
    )

    lines.append(
        f"20DMA: {credit['hyg_lqd_ma20']:.4f} "
        f"({'✓ above' if credit['hyg_lqd_above20'] else '✗ below'})"
    )

    lines.append(
        f"50DMA: {credit['hyg_lqd_ma50']:.4f} "
        f"({'✓ above' if credit['hyg_lqd_above50'] else '✗ below'})"
    )

    lines.append(
        f"20D change: "
        f"{credit['hyg_lqd_20d']:+.2f}%"
    )

    lines.append(
        f"63D change: "
        f"{credit['hyg_lqd_63d']:+.2f}%"
    )

    lines.append("")

    if credit["hyg_lqd_above20"]:

        lines.append(
            "→ The credit market remains "
            "relatively healthy for now."
        )

        lines.append(
            "→ Investors are not yet showing "
            "a mass exit from riskier credit."
        )

    else:

        lines.append(
            "→ The credit structure is starting to deteriorate."
        )

        lines.append(
            "→ This matters because credit often "
            "starts showing stress before "
            "a full risk-off becomes obvious "
            "in the broad equity market."
        )

    # ========================================================
    # MACRO
    # ========================================================

    lines.append("")

    lines.append(
        "5️⃣ MACRO BACKDROP"
    )

    lines.append(
        "──────────────────────────────"
    )

    if not np.isnan(macro["tnx"]):

        lines.append(
            f"10Y Treasury yield: "
            f"{macro['tnx']:.2f}%"
        )

        lines.append(
            f"20-day change: "
            f"{macro['tnx_20d']:+.2f}%"
        )

        if macro["tnx_20d"] < 0:

            lines.append(
                "→ The 10-year Treasury yield "
                "is falling. This is consistent with rising "
                "bond prices."
            )

        else:

            lines.append(
                "→ The 10-year Treasury yield "
                "is rising. This limits "
                "the bullish duration signal."
            )

    if not np.isnan(macro["dxy"]):

        lines.append(
            f"DXY: {macro['dxy']:.2f}"
        )

        lines.append(
            f"DXY 20-day change: "
            f"{macro['dxy_20d']:+.2f}%"
        )

    # ========================================================
    # REGIME
    # ========================================================

    lines.append("")

    lines.append(
        "6️⃣ CURRENT MARKET STATE"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        regime
    )

    lines.append("")

    lines.append(
        f"Duration score: {duration_score}/2"
    )

    lines.append(
        f"Equity score: {equity_score}/3"
    )

    lines.append(
        f"Credit score: {credit_score}/2"
    )

    lines.append("")

    lines.append(
        "Main conclusion:"
    )

    lines.append(
        explanation
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    lines.append("")

    lines.extend(
        build_interpretation(
            duration,
            vix,
            equities,
            credit,
            regime,
        )
    )

    # ========================================================
    # THEORY MAP
    # ========================================================

    lines.append("")

    lines.append(
        "🔟 THEORY SIGNAL MAP"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        "TLT ↑ + VIX ↓ + SPY ↑ + CREDIT ↑"
    )

    lines.append(
        "→ MACRO DISLOCATION"
    )

    lines.append(
        "Bonds are warning, "
        "while the risk market is calm for now."
    )

    lines.append("")

    lines.append(
        "TLT ↑ + SPY ↓ + CREDIT ↓ + VIX ↑"
    )

    lines.append(
        "→ RISK-OFF"
    )

    lines.append(
        "Defensive assets and risk indicators "
        "already confirm each other."
    )

    lines.append("")

    lines.append(
        "SPY ↑ + CREDIT ↑ + VIX ↓ + TLT neutral"
    )

    lines.append(
        "→ RISK-ON / HEALTHY"
    )

    lines.append(
        "Risk markets are consistent with each other."
    )

    lines.append("")

    lines.append(
        "SPY ↑ + CREDIT ↑ + VIX ↓ + TLT ↑"
    )

    lines.append(
        "→ MAIN OBSERVATION ZONE"
    )

    lines.append(
        "This is not an immediate exit signal. "
        "It is a situation where one needs to watch "
        "whether the equity market starts confirming "
        "the bond market's macro signal."
    )

    # ========================================================
    # WHAT TO WATCH NEXT
    # ========================================================

    lines.append("")

    lines.append(
        "1️⃣1️⃣ WHAT TO WATCH NEXT"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        "1. VIX"
    )

    lines.append(
        "If VIX starts rising steadily "
        "from extremely low levels, "
        "that will be the first confirmation "
        "of rising perceived risk."
    )

    lines.append("")

    lines.append(
        "2. HYG/LQD"
    )

    lines.append(
        "If HYG/LQD drops below the 20DMA and then "
        "the 50DMA, credit will start confirming "
        "macroeconomic deterioration."
    )

    lines.append("")

    lines.append(
        "3. SPY"
    )

    lines.append(
        "If SPY loses the 50DMA and then "
        "the 200DMA, the divergence between bonds "
        "and equities will start turning "
        "into a full risk-off."
    )

    lines.append("")

    lines.append(
        "4. TLT"
    )

    lines.append(
        "If TLT keeps strengthening "
        "and TLT/IEF strengthens at the same time, "
        "the signal of strong demand for duration "
        "will become more convincing."
    )

    lines.append("")

    lines.append(
        "5. 10Y YIELD"
    )

    lines.append(
        "A fall in the 10-year Treasury yield "
        "together with a rise in TLT will be "
        "additional confirmation of the move "
        "towards lower yields."
    )

    # ========================================================
    # FINAL
    # ========================================================

    lines.append("")

    lines.append(
        "1️⃣2️⃣ SUMMARY"
    )

    lines.append(
        "──────────────────────────────"
    )

    if regime == "🟠 MACRO DISLOCATION":

        lines.append(
            "The market is NOT in a confirmed "
            "crisis regime."
        )

        lines.append(
            "But there is an important divergence:"
        )

        lines.append(
            "LONG TREASURIES → caution / "
            "shifting macro expectations"
        )

        lines.append(
            "SPY → strong for now"
        )

        lines.append(
            "CREDIT → healthy for now"
        )

        lines.append(
            "VIX → very low for now"
        )

        lines.append("")

        lines.append(
            "Therefore:"
        )

        lines.append(
            "🟠 The situation needs watching, "
            "but by itself is NOT yet "
            "a confirmed bearish signal."
        )

    elif regime == "🔴 RISK-OFF":

        lines.append(
            "The market is in a confirmed "
            "defensive regime."
        )

        lines.append(
            "Several independent indicators "
            "already agree with each other."
        )

    elif regime == "🟢 RISK-ON / HEALTHY":

        lines.append(
            "The market is in a normal "
            "risk-on regime."
        )

        lines.append(
            "The main market segments give no "
            "strong contradictory signal."
        )

    else:

        lines.append(
            "The market is in a transitional "
            "or mixed state."
        )

        lines.append(
            "Need to wait for confirmation "
            "from several independent indicators."
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(lines)


# ============================================================
# SAVE
# ============================================================

def save_report(
    date,
    report,
):

    import os

    os.makedirs(
        "results",
        exist_ok=True,
    )

    file = os.path.join(
        "results",
        "duration_volatility_history.csv",
    )

    row = pd.DataFrame(
        [
            {
                "market_date":
                    date.date().isoformat(),

                "report":
                    report,
            }
        ]
    )

    if os.path.exists(file):

        old = pd.read_csv(
            file
        )

        old = old[
            old["market_date"]
            != date.date().isoformat()
        ]

        row = pd.concat(
            [old, row],
            ignore_index=True,
        )

    row.to_csv(
        file,
        index=False,
    )

    print(
        f"Saved: {file}"
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    report,
):

    try:

        from theories.telegram import (
            send_message
        )

        return send_message(
            report
        )

    except Exception as exc:

        print(
            f"Telegram unavailable: {exc}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def run():

    args = parse_args()

    close = download_data(
        args.period
    )

    date = select_date(
        close,
        args.date
    )

    print()
    print(
        f"Using market date: "
        f"{date.date()}"
    )

    duration = calculate_duration(
        close,
        date
    )

    vix = calculate_vix(
        close,
        date
    )

    equities = calculate_equities(
        close,
        date
    )

    credit = calculate_credit(
        close,
        date
    )

    macro = calculate_macro(
        close,
        date
    )

    (
        regime,
        duration_score,
        equity_score,
        credit_score,
        explanation,
    ) = determine_regime(
        duration,
        vix,
        equities,
        credit,
    )

    report = build_report(
        date,
        duration,
        vix,
        equities,
        credit,
        macro,
        regime,
        duration_score,
        equity_score,
        credit_score,
        explanation,
    )

    print()
    print(report)
    print()

    if args.save:

        save_report(
            date,
            report,
        )

    if args.telegram:

        send_telegram(
            report
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run()

    except KeyboardInterrupt:

        print(
            "\nStopped."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            f"ERROR: {exc}"
        )

        sys.exit(1)

