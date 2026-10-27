import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from theories.telegram import send_message


# ============================================================
# MACRO STRESS / DISLOCATION MONITOR
#
# THIRD THEORY
#
# Idea:
#
# The market can look calm by VIX and equities,
# while at the same time the bond market may be showing
# a completely different macroeconomic signal.
#
# We look for divergence between:
#
#   1. Equities
#   2. Long Treasury bonds
#   3. Short Treasury bonds
#   4. The credit market
#   5. VIX
#   6. The 10-year Treasury yield
#   7. The dollar
#
# MAIN HYPOTHESIS
#
# If:
#
#   TLT is rising
#   TLT outperforms IEF/SHY
#   the 10Y yield is falling
#   VIX stays low
#   SPY stays strong
#   credit stays calm
#
# then a MACRO DISLOCATION may be forming:
#
# the bond market starts pricing in
# an economic slowdown / lower inflation / future
# rate cuts before this shows up in equities.
#
# This is NOT an automatic signal to sell equities.
#
# The theory should show:
#
#   NORMAL RISK-ON
#   HIDDEN TENSION
#   MACRO DISLOCATION
#   MACRO STRESS
#   RISK-OFF
#
# What matters is not one indicator but the consistency
# of several independent market segments.
# ============================================================


TZ = ZoneInfo("America/New_York")

DEFAULT_PERIOD = "5y"

MA20 = 20
MA50 = 50
MA200 = 200


# ============================================================
# TICKERS
# ============================================================

TICKERS = {
    "SPY": "SPY",

    # Long Treasuries
    "TLT": "TLT",

    # Intermediate duration
    "IEF": "IEF",

    # Short duration
    "SHY": "SHY",

    # Credit
    "HYG": "HYG",
    "LQD": "LQD",

    # Volatility
    "VIX": "^VIX",

    # 10Y Treasury yield
    "TNX": "^TNX",

    # Dollar
    "DXY": "DX-Y.NYB",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Macro Stress / Dislocation Monitor"
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Historical date YYYY-MM-DD",
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="Yahoo Finance download period",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the report to CSV",
    )

    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send the report to Telegram",
    )

    return parser.parse_args()


# ============================================================
# DOWNLOAD
# ============================================================

def download_data(period):

    print()
    print("Loading macro data...")
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
            f"Yahoo Finance error: {exc}"
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

        close = data[["Close"]].copy()

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
        f"Period: "
        f"{close.index[0].date()} -> "
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

    if len(dates) == 0:

        raise RuntimeError(
            "No market data."
        )

    if requested:

        target = pd.Timestamp(
            requested
        )

        available = dates[
            dates <= target
        ]

        if len(available) == 0:

            raise RuntimeError(
                "No data before the given date."
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

    series = series.dropna()

    if len(series) <= days:

        return np.nan

    current = series.iloc[-1]
    previous = series.iloc[-days - 1]

    if pd.isna(current) or pd.isna(previous):

        return np.nan

    if previous == 0:

        return np.nan

    return (
        current / previous - 1
    ) * 100


def moving_average(
    series,
    window,
):

    series = series.dropna()

    if len(series) < window:

        return np.nan

    return float(
        series.tail(window).mean()
    )


def safe_bool(value):

    if pd.isna(value):

        return False

    return bool(value)


# ============================================================
# TREASURY DURATION
# ============================================================

def calculate_duration(
    close,
    date,
):

    data = close.loc[:date]

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

    # --------------------------------------------------------
    # TLT / IEF
    # --------------------------------------------------------

    tlt_ief = (
        tlt / ief
    ).dropna()

    ratio_ief = float(
        tlt_ief.iloc[-1]
    )

    ratio_ief_ma20 = moving_average(
        tlt_ief,
        MA20,
    )

    ratio_ief_ma50 = moving_average(
        tlt_ief,
        MA50,
    )

    ratio_ief_63d = pct_change(
        tlt_ief,
        63,
    )

    # --------------------------------------------------------
    # TLT / SHY
    # --------------------------------------------------------

    tlt_shy = (
        tlt / shy
    ).dropna()

    ratio_shy = float(
        tlt_shy.iloc[-1]
    )

    ratio_shy_ma20 = moving_average(
        tlt_shy,
        MA20,
    )

    ratio_shy_63d = pct_change(
        tlt_shy,
        63,
    )

    return {

        "tlt": tlt_current,

        "tlt_ma20": tlt_ma20,
        "tlt_ma50": tlt_ma50,
        "tlt_ma200": tlt_ma200,

        "tlt_20d": tlt_20d,
        "tlt_63d": tlt_63d,
        "tlt_252d": tlt_252d,

        "tlt_above20":
            tlt_current > tlt_ma20,

        "tlt_above50":
            tlt_current > tlt_ma50,

        "tlt_above200":
            tlt_current > tlt_ma200,

        "tlt_ief":
            ratio_ief,

        "tlt_ief_ma20":
            ratio_ief_ma20,

        "tlt_ief_ma50":
            ratio_ief_ma50,

        "tlt_ief_63d":
            ratio_ief_63d,

        "tlt_ief_above20":
            ratio_ief > ratio_ief_ma20,

        "tlt_ief_above50":
            ratio_ief > ratio_ief_ma50,

        "tlt_shy":
            ratio_shy,

        "tlt_shy_ma20":
            ratio_shy_ma20,

        "tlt_shy_63d":
            ratio_shy_63d,

        "tlt_shy_above20":
            ratio_shy > ratio_shy_ma20,
    }


# ============================================================
# VIX
# ============================================================

def calculate_vix(
    close,
    date,
):

    vix = close.loc[
        :date,
        "VIX"
    ].dropna()

    if len(vix) < MA50:

        raise RuntimeError(
            "Not enough VIX history."
        )

    current = float(
        vix.iloc[-1]
    )

    ma20 = moving_average(
        vix,
        MA20,
    )

    ma50 = moving_average(
        vix,
        MA50,
    )

    percentile_data = vix.tail(
        252
    )

    percentile = (
        percentile_data < current
    ).mean() * 100

    change_20d = pct_change(
        vix,
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

        "change20":
            change_20d,

        "below20":
            current < ma20,

        "below50":
            current < ma50,

        "very_low":
            current < 15,

        "low":
            current < 20,

        "elevated":
            current >= 20,
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

    if len(spy) < MA200:

        raise RuntimeError(
            "Not enough SPY history."
        )

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

    data = close.loc[:date]

    hyg = data["HYG"].dropna()
    lqd = data["LQD"].dropna()

    if len(hyg) < MA50:

        raise RuntimeError(
            "Not enough HYG history."
        )

    ratio = (
        hyg / lqd
    ).dropna()

    current = float(
        ratio.iloc[-1]
    )

    ma20 = moving_average(
        ratio,
        MA20,
    )

    ma50 = moving_average(
        ratio,
        MA50,
    )

    return {

        "hyg":
            float(hyg.iloc[-1]),

        "lqd":
            float(lqd.iloc[-1]),

        "ratio":
            current,

        "ma20":
            ma20,

        "ma50":
            ma50,

        "above20":
            current > ma20,

        "above50":
            current > ma50,

        "return20":
            pct_change(
                ratio,
                20,
            ),

        "return63":
            pct_change(
                ratio,
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

        current = float(
            tnx.iloc[-1]
        )

        result["tnx"] = current

        result["tnx20d"] = pct_change(
            tnx,
            20,
        )

        result["tnx_below20"] = (
            current
            < moving_average(
                tnx,
                MA20,
            )
        )

    else:

        result["tnx"] = np.nan
        result["tnx20d"] = np.nan
        result["tnx_below20"] = False

    if len(dxy) > 20:

        current = float(
            dxy.iloc[-1]
        )

        result["dxy"] = current

        result["dxy20d"] = pct_change(
            dxy,
            20,
        )

        result["dxy_below20"] = (
            current
            < moving_average(
                dxy,
                MA20,
            )
        )

    else:

        result["dxy"] = np.nan
        result["dxy20d"] = np.nan
        result["dxy_below20"] = False

    return result


# ============================================================
# REGIME ANALYSIS
# ============================================================

def determine_regime(
    duration,
    vix,
    equities,
    credit,
    macro,
):

    # ========================================================
    # 1. DURATION
    # ========================================================

    duration_strong = (
        duration["tlt_above50"]
        and duration["tlt_63d"] > 0
    )

    duration_relative = (
        duration["tlt_ief_above20"]
        and duration["tlt_ief_63d"] > 0
    )

    duration_long_vs_short = (
        duration["tlt_shy_above20"]
        and duration["tlt_shy_63d"] > 0
    )

    duration_score = sum(
        [
            duration_strong,
            duration_relative,
            duration_long_vs_short,
        ]
    )

    # ========================================================
    # 2. VIX
    # ========================================================

    vix_complacent = (
        vix["current"] < 15
    )

    vix_low = (
        vix["current"] < 20
    )

    # ========================================================
    # 3. EQUITIES
    # ========================================================

    equity_strong = (
        equities["above50"]
        and equities["above200"]
    )

    equity_weak = (
        not equities["above50"]
    )

    equity_score = sum(
        [
            equities["above20"],
            equities["above50"],
            equities["above200"],
        ]
    )

    # ========================================================
    # 4. CREDIT
    # ========================================================

    credit_healthy = (
        credit["above20"]
        and credit["above50"]
    )

    credit_warning = (
        not credit["above20"]
        or not credit["above50"]
    )

    credit_score = sum(
        [
            credit["above20"],
            credit["above50"],
        ]
    )

    # ========================================================
    # 5. BOND YIELD
    # ========================================================

    falling_yield = False

    if not np.isnan(
        macro["tnx20d"]
    ):

        falling_yield = (
            macro["tnx20d"] < 0
        )

    # ========================================================
    # 6. MACRO DISLOCATION
    #
    # Bonds are saying:
    #
    # "Future growth/inflation may be weaker."
    #
    # Equities are saying:
    #
    # "Everything is fine."
    #
    # VIX is saying:
    #
    # "No immediate danger."
    #
    # Credit is saying:
    #
    # "No stress."
    #
    # This disagreement is the main object
    # of this theory.
    # ========================================================

    dislocation_points = sum(
        [
            duration_strong,
            duration_relative,
            falling_yield,
            vix_complacent,
            equity_strong,
            credit_healthy,
        ]
    )

    # ========================================================
    # RISK-OFF
    # ========================================================

    if (
        duration_score >= 2
        and equity_weak
        and credit_warning
        and vix["elevated"]
    ):

        regime = "RISK-OFF"

        explanation = (
            "Markets are in a synchronised "
            "defensive move. Long Treasuries "
            "are strengthening, equities are weakening, credit "
            "is deteriorating and VIX is elevated. This is no longer "
            "just a divergence of signals but a full "
            "repricing of risk."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            dislocation_points,
            explanation,
        )

    # ========================================================
    # MACRO STRESS
    # ========================================================

    if (
        duration_score >= 2
        and equity_weak
        and (
            credit_warning
            or falling_yield
        )
    ):

        regime = "MACRO STRESS"

        explanation = (
            "The bond market shows a defensive "
            "signal together with weakening equities. "
            "This means market participants are starting "
            "to price in a deteriorating macroeconomic "
            "situation. If credit and VIX also deteriorate, "
            "the regime may turn into a full RISK-OFF."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            dislocation_points,
            explanation,
        )

    # ========================================================
    # MACRO DISLOCATION
    # ========================================================

    if (
        duration_score >= 2
        and falling_yield
        and equity_strong
        and vix_complacent
        and credit_healthy
    ):

        regime = "MACRO DISLOCATION"

        explanation = (
            "This is the most interesting divergence. "
            "Equities are strong, VIX "
            "stays very low, credit shows no "
            "stress, but long Treasuries are at the same time "
            "strengthening and outperforming shorter "
            "duration. A falling 10-year "
            "yield reinforces this signal. "
            "In other words, the equity market looks calm, "
            "while the bond market starts pricing in "
            "weaker future growth, lower inflationary "
            "pressure or a potentially softer "
            "monetary policy."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            dislocation_points,
            explanation,
        )

    # ========================================================
    # DURATION WARNING
    # ========================================================

    if (
        duration_score >= 2
        and equity_strong
    ):

        regime = "DURATION WARNING"

        explanation = (
            "Long Treasuries show strength, while "
            "equities remain strong for now. This is not "
            "a crisis signal yet. However, the bond "
            "market is starting to diverge from equities. "
            "Watch whether the fall in the "
            "10Y yield deepens and whether TLT keeps "
            "outperforming IEF and SHY."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            dislocation_points,
            explanation,
        )

    # ========================================================
    # COMPLACENCY
    # ========================================================

    if (
        vix_complacent
        and equity_strong
        and credit_healthy
    ):

        regime = "COMPRESSION / COMPLACENCY"

        explanation = (
            "All main risk assets are in a "
            "comfortable state. Equities are strong, credit "
            "is healthy and VIX is very low. There is no "
            "confirmed macro stress yet. However, a low "
            "VIX means the market prices in practically no "
            "significant short-term "
            "swings."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            dislocation_points,
            explanation,
        )

    # ========================================================
    # HEALTHY RISK-ON
    # ========================================================

    if (
        equity_strong
        and credit_healthy
        and duration_score < 2
    ):

        regime = "RISK-ON / HEALTHY"

        explanation = (
            "Equities and credit move consistently. "
            "There is no strong signal from long "
            "Treasuries that would contradict the behaviour "
            "of equities. This is the most normal configuration "
            "risk-on."
        )

        return (
            regime,
            duration_score,
            equity_score,
            credit_score,
            dislocation_points,
            explanation,
        )

    # ========================================================
    # MIXED
    # ========================================================

    regime = "MIXED"

    explanation = (
        "The market is not in one clean regime. "
        "Some indicators point to risk-on, some "
        "to a defensive move or macroeconomic "
        "uncertainty. In such a situation a single "
        "signal cannot be treated as a standalone "
        "trading signal."
    )

    return (
        regime,
        duration_score,
        equity_score,
        credit_score,
        dislocation_points,
        explanation,
    )


# ============================================================
# INTERPRETATION ENGINE
# ============================================================

def build_interpretation(
    duration,
    vix,
    equities,
    credit,
    macro,
    regime,
):

    lines = []

    lines.append(
        "MARKET INTERPRETATION"
    )

    lines.append(
        "------------------------------"
    )

    # --------------------------------------------------------
    # BONDS
    # --------------------------------------------------------

    if (
        duration["tlt_63d"] > 0
        and duration["tlt_ief_63d"] > 0
    ):

        lines.append(
            "1. BONDS:"
        )

        lines.append(
            "Long Treasuries are strengthening and "
            "outperforming intermediate duration."
        )

        lines.append(
            "Implication: the bond market allows for "
            "a change in future macro conditions."
        )

    elif duration["tlt_63d"] > 0:

        lines.append(
            "1. BONDS:"
        )

        lines.append(
            "TLT is rising, but the relative advantage "
            "of long duration is not strong enough yet."
        )

    else:

        lines.append(
            "1. BONDS:"
        )

        lines.append(
            "TLT shows no steady rise."
        )

    # --------------------------------------------------------
    # YIELDS
    # --------------------------------------------------------

    if not np.isnan(
        macro["tnx20d"]
    ):

        lines.append(
            "2. 10Y YIELD:"
        )

        if macro["tnx20d"] < 0:

            lines.append(
                "The 10-year Treasury yield "
                "is falling."
            )

            lines.append(
                "Implication: bond prices get "
                "support, and the market starts pricing in "
                "lower future yields."
            )

        else:

            lines.append(
                "The 10-year Treasury yield "
                "is not falling."
            )

    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    lines.append(
        "3. VIX:"
    )

    if vix["very_low"]:

        lines.append(
            "VIX is in the very low "
            "volatility zone."
        )

        lines.append(
            "Implication: the equity market does not expect "
            "serious short-term stress."
        )

    elif vix["low"]:

        lines.append(
            "VIX is in a relatively calm zone."
        )

    else:

        lines.append(
            "VIX is elevated."
        )

        lines.append(
            "Implication: market participants are starting "
            "to pay more for volatility protection."
        )

    # --------------------------------------------------------
    # EQUITIES
    # --------------------------------------------------------

    lines.append(
        "4. EQUITIES:"
    )

    if equities["above50"] and equities["above200"]:

        lines.append(
            "SPY is above the 50DMA and 200DMA."
        )

        lines.append(
            "Implication: the medium- and long-term "
            "trend remains constructive."
        )

    elif equities["above200"]:

        lines.append(
            "SPY is above the 200DMA, but the structure "
            "is weaker in the short term."
        )

    else:

        lines.append(
            "SPY is below the 200DMA."
        )

        lines.append(
            "Implication: the long-term structure "
            "of equities has deteriorated."
        )

    # --------------------------------------------------------
    # CREDIT
    # --------------------------------------------------------

    lines.append(
        "5. CREDIT:"
    )

    if credit["above20"] and credit["above50"]:

        lines.append(
            "HYG/LQD is above the 20DMA and 50DMA."
        )

        lines.append(
            "Implication: the high-yield credit market "
            "shows no pronounced stress yet."
        )

    else:

        lines.append(
            "HYG/LQD has weakened."
        )

        lines.append(
            "Implication: the credit market is starting "
            "to confirm rising risk."
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "6. MAIN CONCLUSION:"
    )

    if regime == "MACRO DISLOCATION":

        lines.append(
            "A divergence is forming between the equity "
            "market and the bond market."
        )

        lines.append(
            "Equities still behave as in a normal "
            "risk-on, but Treasuries are starting to price in "
            "a change in the future macroeconomic environment."
        )

        lines.append(
            "This does NOT mean an immediate fall in equities."
        )

        lines.append(
            "It means the bond market "
            "is sending a warning signal that "
            "should be checked via credit and VIX."
        )

    elif regime == "DURATION WARNING":

        lines.append(
            "There is an early divergence between bonds "
            "and equities."
        )

        lines.append(
            "For now this is a warning, not confirmed "
            "stress."
        )

    elif regime == "MACRO STRESS":

        lines.append(
            "Macroeconomic tension is already starting "
            "to show not only in bonds but also "
            "in equities or credit."
        )

        lines.append(
            "If VIX keeps rising and credit keeps deteriorating, "
            "the probability of a transition to RISK-OFF increases."
        )

    elif regime == "RISK-OFF":

        lines.append(
            "The main market segments are moving in sync "
            "into a defensive regime."
        )

        lines.append(
            "This is the most serious configuration "
            "considered by this theory."
        )

    elif regime == "RISK-ON / HEALTHY":

        lines.append(
            "No substantial macro divergence yet."
        )

        lines.append(
            "Equities and credit confirm a normal "
            "risk-on regime."
        )

    elif regime == "COMPRESSION / COMPLACENCY":

        lines.append(
            "The market is calm and complacent."
        )

        lines.append(
            "The main risk is not confirmed now, "
            "but a very low VIX means a low "
            "price of protection against an unexpected event."
        )

    else:

        lines.append(
            "The market is in a transitional or "
            "uncertain state."
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
    dislocation_points,
    explanation,
):

    lines = []

    lines.append(
        "MACRO STRESS / DISLOCATION MONITOR"
    )

    lines.append(
        "======================================"
    )

    lines.append(
        f"Market date: {date.date()}"
    )

    lines.append("")

    # ========================================================
    # 1. TREASURIES
    # ========================================================

    lines.append(
        "1. LONG TREASURY BONDS"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append(
        f"TLT: {duration['tlt']:.2f}"
    )

    lines.append(
        f"20DMA: {duration['tlt_ma20']:.2f} "
        f"{'OK' if duration['tlt_above20'] else 'NO'}"
    )

    lines.append(
        f"50DMA: {duration['tlt_ma50']:.2f} "
        f"{'OK' if duration['tlt_above50'] else 'NO'}"
    )

    lines.append(
        f"200DMA: {duration['tlt_ma200']:.2f} "
        f"{'OK' if duration['tlt_above200'] else 'NO'}"
    )

    lines.append(
        f"20-day change: "
        f"{duration['tlt_20d']:+.2f}%"
    )

    lines.append(
        f"63-day change: "
        f"{duration['tlt_63d']:+.2f}%"
    )

    lines.append(
        f"252-day change: "
        f"{duration['tlt_252d']:+.2f}%"
    )

    lines.append("")

    lines.append(
        "TLT / IEF:"
    )

    lines.append(
        f"Current ratio: "
        f"{duration['tlt_ief']:.4f}"
    )

    lines.append(
        f"20DMA: "
        f"{duration['tlt_ief_ma20']:.4f} "
        f"{'OK' if duration['tlt_ief_above20'] else 'NO'}"
    )

    lines.append(
        f"50DMA: "
        f"{duration['tlt_ief_ma50']:.4f} "
        f"{'OK' if duration['tlt_ief_above50'] else 'NO'}"
    )

    lines.append(
        f"63D relative change: "
        f"{duration['tlt_ief_63d']:+.2f}%"
    )

    lines.append("")

    lines.append(
        "TLT / SHY:"
    )

    lines.append(
        f"63D relative change: "
        f"{duration['tlt_shy_63d']:+.2f}%"
    )

    lines.append("")

    # ========================================================
    # 2. VIX
    # ========================================================

    lines.append(
        "2. VOLATILITY"
    )

    lines.append(
        "--------------------------------------"
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
        f"252D percentile: "
        f"{vix['percentile']:.1f}%"
    )

    lines.append(
        f"20-day change: "
        f"{vix['change20']:+.2f}%"
    )

    if vix["very_low"]:

        lines.append(
            "VIX regime: VERY LOW"
        )

    elif vix["low"]:

        lines.append(
            "VIX regime: LOW"
        )

    elif vix["current"] < 30:

        lines.append(
            "VIX regime: ELEVATED"
        )

    else:

        lines.append(
            "VIX regime: HIGH"
        )

    lines.append("")

    # ========================================================
    # 3. EQUITIES
    # ========================================================

    lines.append(
        "3. EQUITIES"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append(
        f"SPY: {equities['spy']:.2f}"
    )

    lines.append(
        f"20DMA: {equities['ma20']:.2f} "
        f"{'OK' if equities['above20'] else 'NO'}"
    )

    lines.append(
        f"50DMA: {equities['ma50']:.2f} "
        f"{'OK' if equities['above50'] else 'NO'}"
    )

    lines.append(
        f"200DMA: {equities['ma200']:.2f} "
        f"{'OK' if equities['above200'] else 'NO'}"
    )

    lines.append(
        f"20D return: "
        f"{equities['return20']:+.2f}%"
    )

    lines.append(
        f"63D return: "
        f"{equities['return63']:+.2f}%"
    )

    lines.append(
        f"252D return: "
        f"{equities['return252']:+.2f}%"
    )

    lines.append("")

    # ========================================================
    # 4. CREDIT
    # ========================================================

    lines.append(
        "4. CREDIT MARKET"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append(
        f"HYG: {credit['hyg']:.2f}"
    )

    lines.append(
        f"LQD: {credit['lqd']:.2f}"
    )

    lines.append(
        f"HYG/LQD: {credit['ratio']:.4f}"
    )

    lines.append(
        f"20DMA: {credit['ma20']:.4f} "
        f"{'OK' if credit['above20'] else 'NO'}"
    )

    lines.append(
        f"50DMA: {credit['ma50']:.4f} "
        f"{'OK' if credit['above50'] else 'NO'}"
    )

    lines.append(
        f"20D change: "
        f"{credit['return20']:+.2f}%"
    )

    lines.append(
        f"63D change: "
        f"{credit['return63']:+.2f}%"
    )

    lines.append("")

    # ========================================================
    # 5. MACRO
    # ========================================================

    lines.append(
        "5. MACROECONOMIC INDICATORS"
    )

    lines.append(
        "--------------------------------------"
    )

    if not np.isnan(
        macro["tnx"]
    ):

        lines.append(
            f"10Y Treasury yield: "
            f"{macro['tnx']:.2f}%"
        )

        lines.append(
            f"10Y 20-day change: "
            f"{macro['tnx20d']:+.2f}%"
        )

        if macro["tnx20d"] < 0:

            lines.append(
                "The 10Y yield is falling."
            )

            lines.append(
                "This supports the hypothesis of "
                "rising duration demand."
            )

        else:

            lines.append(
                "The 10Y yield is not falling."
            )

    if not np.isnan(
        macro["dxy"]
    ):

        lines.append(
            f"DXY: {macro['dxy']:.2f}"
        )

        lines.append(
            f"DXY 20-day change: "
            f"{macro['dxy20d']:+.2f}%"
        )

    lines.append("")

    # ========================================================
    # 6. REGIME
    # ========================================================

    lines.append(
        "6. CURRENT MARKET STATE"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append(
        regime
    )

    lines.append("")

    lines.append(
        f"Duration score: "
        f"{duration_score}/3"
    )

    lines.append(
        f"Equity score: "
        f"{equity_score}/3"
    )

    lines.append(
        f"Credit score: "
        f"{credit_score}/2"
    )

    lines.append(
        f"Macro dislocation points: "
        f"{dislocation_points}/6"
    )

    lines.append("")

    lines.append(
        "MAIN CONCLUSION:"
    )

    lines.append(
        explanation
    )

    lines.append("")

    # ========================================================
    # 7. DETAILED INTERPRETATION
    # ========================================================

    interpretation = build_interpretation(
        duration,
        vix,
        equities,
        credit,
        macro,
        regime,
    )

    lines.extend(
        interpretation
    )

    lines.append("")

    # ========================================================
    # 8. HOW TO READ THE THEORY
    # ========================================================

    lines.append(
        "7. HOW TO READ THIS THEORY"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append(
        "Normal risk-on:"
    )

    lines.append(
        "SPY rising + credit healthy + VIX low "
        "+ TLT shows no strong outperformance."
    )

    lines.append("")

    lines.append(
        "Early warning:"
    )

    lines.append(
        "TLT starts rising and outperforming IEF/SHY, "
        "but SPY and credit are still strong."
    )

    lines.append("")

    lines.append(
        "MACRO DISLOCATION:"
    )

    lines.append(
        "TLT rising + 10Y yield falling "
        "+ SPY stays strong + VIX low "
        "+ credit healthy."
    )

    lines.append("")

    lines.append(
        "MACRO STRESS:"
    )

    lines.append(
        "Treasuries strong + equities start weakening "
        "+ credit deteriorating."
    )

    lines.append("")

    lines.append(
        "RISK-OFF:"
    )

    lines.append(
        "Treasuries strong + equities weak "
        "+ credit weak + VIX high."
    )

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "MACRO DISLOCATION does not automatically mean "
        "a market decline."
    )

    lines.append(
        "It is a signal that the bond market "
        "may be seeing a change in future economic "
        "conditions before the equity market."
    )

    lines.append("")

    lines.append(
        "The main question after a dislocation is detected:"
    )

    lines.append(
        "Will credit and VIX confirm this move?"
    )

    lines.append("")

    lines.append(
        "If credit stays strong and VIX low, "
        "it may simply be an expectation of lower "
        "rates or inflation."
    )

    lines.append("")

    lines.append(
        "If credit starts falling and VIX rises, "
        "the divergence turns into a more serious "
        "MACRO STRESS."
    )

    lines.append("")

    lines.append(
        "If SPY then loses the 50DMA and then "
        "the 200DMA, the hypothesis gets confirmation "
        "from equities as well."
    )

    lines.append("")

    lines.append(
        "======================================"
    )

    return "\n".join(lines)


# ============================================================
# SAVE
# ============================================================

def save_report(
    date,
    report,
):

    os.makedirs(
        "results",
        exist_ok=True,
    )

    file = os.path.join(
        "results",
        "macro_stress_history.csv",
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
            [
                old,
                row,
            ],
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
        date,
    )

    vix = calculate_vix(
        close,
        date,
    )

    equities = calculate_equities(
        close,
        date,
    )

    credit = calculate_credit(
        close,
        date,
    )

    macro = calculate_macro(
        close,
        date,
    )

    (
        regime,
        duration_score,
        equity_score,
        credit_score,
        dislocation_points,
        explanation,
    ) = determine_regime(
        duration,
        vix,
        equities,
        credit,
        macro,
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
        dislocation_points,
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

        send_message(
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