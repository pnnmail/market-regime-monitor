import argparse
import os
import sys
from io import StringIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from theories.database import (
    initialize_database,
    get_last_market_date,
    get_database_stats,
    load_prices,
    save_prices,
    save_analysis,
)

from theories.telegram import send_message


TZ = ZoneInfo("America/New_York")

DATA_DIR = "data"

CORR_DAYS = 63
MA20 = 20
MA50 = 50
MA200 = 200

DEFAULT_HISTORY_YEARS = 3
DEFAULT_BATCH_SIZE = 75

BREADTH_WEAK = 55
BREADTH_RECOVERY = 60
BREADTH_CONSTRUCTIVE = 65
BREADTH_NEAR_CONFIRMATION = 68
BREADTH_CONFIRMED = 72


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="RSP Breadth Hypothesis Monitor"
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Historical date YYYY-MM-DD",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save analysis to SQLite",
    )

    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send report to Telegram",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force full historical download",
    )

    parser.add_argument(
        "--history-years",
        type=int,
        default=DEFAULT_HISTORY_YEARS,
        help="Initial historical period",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Yahoo Finance batch size",
    )

    return parser.parse_args()


# ============================================================
# S&P 500
# ============================================================

def get_sp500_symbols():

    print("Loading S&P 500 components...")

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
    }

    cache_file = os.path.join(
        DATA_DIR,
        "sp500_symbols.txt",
    )

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        # newer pandas does not accept an HTML string directly: it is treated as a file path
        tables = pd.read_html(
            StringIO(response.text)
        )

        df = tables[0]

        symbols = (
            df["Symbol"]
            .astype(str)
            .str.strip()
            .str.replace(
                ".",
                "-",
                regex=False,
            )
            .tolist()
        )

        symbols = sorted(
            set(symbols)
        )

        if len(symbols) < 450:

            raise RuntimeError(
                f"Only {len(symbols)} symbols loaded."
            )

        os.makedirs(
            DATA_DIR,
            exist_ok=True,
        )

        with open(
            cache_file,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "\n".join(symbols)
            )

        print(
            f"S&P 500 components: {len(symbols)}"
        )

        return symbols

    except Exception as exc:

        print(
            f"WARNING: S&P 500 download failed: {str(exc)[:200]}"
        )

        if os.path.exists(cache_file):

            with open(
                cache_file,
                "r",
                encoding="utf-8",
            ) as f:

                symbols = [
                    x.strip()
                    for x in f
                    if x.strip()
                ]

            print(
                f"Using cached S&P 500 list: "
                f"{len(symbols)} symbols"
            )

            return symbols

        raise


# ============================================================
# YAHOO DOWNLOAD
# ============================================================

def download_batch(
    symbols,
    start,
    end,
):

    try:

        data = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )

    except Exception as exc:

        print(
            f"Yahoo error: {exc}"
        )

        return pd.DataFrame()

    if data.empty:

        return pd.DataFrame()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if "Close" not in data.columns.get_level_values(0):

            return pd.DataFrame()

        close = data["Close"]

    else:

        if "Close" not in data.columns:

            return pd.DataFrame()

        close = data[
            ["Close"]
        ].copy()

        if len(symbols) == 1:

            close.columns = symbols

    close = close.copy()

    close.columns = [
        str(x).replace(".", "-")
        for x in close.columns
    ]

    close.index = pd.to_datetime(
        close.index
    )

    return close


def download_incremental(
    symbols,
    args,
):

    initialize_database()

    stats = get_database_stats()

    last_date = get_last_market_date()

    today = datetime.now(TZ).date()

    end_date = today + timedelta(days=1)

    # --------------------------------------------------------
    # First run
    # --------------------------------------------------------

    if (
        args.force_download
        or last_date is None
        or stats["dates"] < MA200
    ):

        start_date = (
            today
            - timedelta(
                days=365 * args.history_years
            )
        )

        print()
        print(
            "INITIAL MARKET DOWNLOAD"
        )

        print(
            f"Start: {start_date}"
        )

        print(
            f"End:   {end_date}"
        )

        print(
            f"Symbols: {len(symbols) + 2}"
        )

        download_symbols = (
            symbols
            + ["SPY", "RSP"]
        )

    else:

        last = pd.Timestamp(
            last_date
        ).date()

        start_date = (
            last
            + timedelta(days=1)
        )

        # ----------------------------------------------------
        # Nothing new
        # ----------------------------------------------------

        if start_date > today:

            print()
            print(
                "DATABASE IS UP TO DATE."
            )

            print(
                f"Last market date: {last_date}"
            )

            return load_prices()

        print()
        print(
            "INCREMENTAL MARKET UPDATE"
        )

        print(
            f"Last database date: {last_date}"
        )

        print(
            f"Downloading from:    {start_date}"
        )

        print(
            f"Downloading to:      {end_date}"
        )

        print(
            f"Symbols: {len(symbols) + 2}"
        )

        download_symbols = (
            symbols
            + ["SPY", "RSP"]
        )

    batches = [
        download_symbols[
            i:i + args.batch_size
        ]
        for i in range(
            0,
            len(download_symbols),
            args.batch_size,
        )
    ]

    all_data = []

    for index, batch in enumerate(
        batches,
        start=1,
    ):

        first = (
            (index - 1)
            * args.batch_size
            + 1
        )

        last = min(
            index * args.batch_size,
            len(download_symbols),
        )

        print(
            f"Batch {first}-{last}/"
            f"{len(download_symbols)}"
        )

        data = download_batch(
            batch,
            start=str(start_date),
            end=str(end_date),
        )

        if not data.empty:

            all_data.append(data)

    if all_data:

        new_data = pd.concat(
            all_data,
            axis=1,
        )

        new_data = new_data.loc[
            :,
            ~new_data.columns.duplicated()
        ]

        save_prices(
            new_data
        )

        print(
            "Incremental update saved."
        )

    else:

        print(
            "No new market rows downloaded."
        )

    close = load_prices()

    stats = get_database_stats()

    print()
    print(
        "DATABASE:"
    )

    print(
        f"Dates:   {stats['dates']}"
    )

    print(
        f"Symbols: {stats['symbols']}"
    )

    print(
        f"Rows:    {stats['rows']:,}"
    )

    print(
        f"Range:   "
        f"{stats['first_date']} → "
        f"{stats['last_date']}"
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
                "No data for requested date."
            )

        return available[-1]

    return dates[-1]


# ============================================================
# RSP / SPY
# ============================================================

def calculate_rsp_spy(
    close,
    date,
):

    data = close.loc[
        :date,
        ["RSP", "SPY"],
    ].dropna()

    if len(data) < MA50:

        raise RuntimeError(
            "Not enough RSP/SPY history."
        )

    ratio = (
        data["RSP"]
        / data["SPY"]
    )

    current = float(
        ratio.iloc[-1]
    )

    previous = float(
        ratio.iloc[-2]
    )

    daily_change = (
        current / previous - 1
    ) * 100

    ma20 = float(
        ratio.tail(20).mean()
    )

    ma50 = float(
        ratio.tail(50).mean()
    )

    relative_high_20 = (
        current
        >= ratio.tail(20).max()
    )

    relative_high_63 = (
        current
        >= ratio.tail(63).max()
    )

    relative_high_252 = (
        current
        >= ratio.tail(252).max()
    )

    return {

        "ratio": current,

        "daily_change":
            daily_change,

        "ma20":
            ma20,

        "ma50":
            ma50,

        "above_ma20":
            current > ma20,

        "above_ma50":
            current > ma50,

        "ma20_above_ma50":
            ma20 > ma50,

        "relative_high_20":
            bool(relative_high_20),

        "relative_high_63":
            bool(relative_high_63),

        "relative_high_252":
            bool(relative_high_252),
    }


def classify_rsp(rsp):

    if rsp["relative_high_252"]:

        return "🟢 STRONG"

    if (
        rsp["ma20_above_ma50"]
        and rsp["above_ma20"]
    ):

        return "🟢 CONFIRMED TREND"

    if rsp["above_ma20"]:

        return "🟢 IMPROVING"

    if rsp["above_ma50"]:

        return "🟡 RECOVERY"

    return "🔴 WEAK"


# ============================================================
# BREADTH
# ============================================================

def calculate_breadth(
    close,
    date,
):

    excluded = {
        "SPY",
        "RSP",
        "VTI",
    }

    symbols = [
        x
        for x in close.columns
        if x not in excluded
    ]

    prices = close.loc[
        :date,
        symbols,
    ]

    if len(prices) < MA200:

        raise RuntimeError(
            "Not enough history for 200DMA."
        )

    ma50 = prices.rolling(
        MA50,
        min_periods=MA50,
    ).mean()

    ma200 = prices.rolling(
        MA200,
        min_periods=MA200,
    ).mean()

    current = prices.iloc[-1]

    current50 = ma50.iloc[-1]

    current200 = ma200.iloc[-1]

    valid50 = (
        current.notna()
        & current50.notna()
    )

    valid200 = (
        current.notna()
        & current200.notna()
    )

    above50 = (
        current[valid50]
        > current50[valid50]
    )

    above200 = (
        current[valid200]
        > current200[valid200]
    )

    breadth50 = (
        above50.mean()
        * 100
    )

    breadth200 = (
        above200.mean()
        * 100
    )

    return {

        "breadth50":
            float(breadth50),

        "breadth200":
            float(breadth200),

        "stocks50":
            int(above50.sum()),

        "stocks200":
            int(above200.sum()),

        "valid50":
            int(valid50.sum()),

        "valid200":
            int(valid200.sum()),

        "regime":
            classify_breadth(
                breadth200
            ),
    }


def classify_breadth(
    value
):

    if value < BREADTH_WEAK:

        return "🔴 WEAK"

    if value < BREADTH_RECOVERY:

        return "🟡 RECOVERY"

    if value < BREADTH_CONSTRUCTIVE:

        return "🟡 CONSTRUCTIVE"

    if value < BREADTH_NEAR_CONFIRMATION:

        return "🟢 CONSTRUCTIVE+"

    if value < BREADTH_CONFIRMED:

        return "🟢 NEAR CONFIRMATION"

    return "🟢 CONFIRMED"


# ============================================================
# CORRELATION
# ============================================================

def calculate_correlation(
    close,
    date,
):

    excluded = {
        "SPY",
        "RSP",
        "VTI",
    }

    symbols = [
        x
        for x in close.columns
        if x not in excluded
    ]

    prices = close.loc[
        :date,
        symbols,
    ]

    returns = prices.pct_change(
        fill_method=None
    )

    recent = returns.tail(
        CORR_DAYS
    )

    minimum = int(
        CORR_DAYS * 0.80
    )

    recent = recent.dropna(
        axis=1,
        thresh=minimum,
    )

    if recent.shape[1] < 2:

        raise RuntimeError(
            "Not enough stocks for correlation."
        )

    corr = recent.corr()

    matrix = corr.to_numpy()

    upper = matrix[
        np.triu_indices(
            matrix.shape[0],
            k=1,
        )
    ]

    upper = upper[
        np.isfinite(upper)
    ]

    average = float(
        np.mean(upper)
    )

    median = float(
        np.median(upper)
    )

    if average < 0.10:

        regime = "🟢 VERY LOW"

    elif average < 0.20:

        regime = "🟢 LOW"

    elif average < 0.35:

        regime = "🟡 NORMAL"

    else:

        regime = "🔴 HIGH"

    return {

        "average":
            average,

        "median":
            median,

        "stocks":
            int(recent.shape[1]),

        "pairs":
            int(len(upper)),

        "regime":
            regime,
    }


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    breadth,
    correlation,
    rsp,
):

    conditions = {

        "low_corr":
            correlation["average"] < 0.20,

        "breadth_confirmed":
            breadth["breadth200"] >= 68,

        "rsp_above_20":
            rsp["above_ma20"],

        "rsp_trend":
            rsp["ma20_above_ma50"],
    }

    score = sum(
        conditions.values()
    )

    # --------------------------------------------------------
    # Strong
    # --------------------------------------------------------

    if (
        correlation["average"] < 0.20
        and breadth["breadth200"] >= 72
        and rsp["relative_high_63"]
    ):

        return (
            "🟢 STRONG CONFIRMATION",
            4,
            "Broad market confirmed, "
            "correlation is low, RSP shows "
            "strong relative momentum."
        )

    # --------------------------------------------------------
    # Confirmed
    # --------------------------------------------------------

    if (
        correlation["average"] < 0.20
        and breadth["breadth200"] >= 68
        and rsp["above_ma20"]
    ):

        return (
            "🟢 BULLISH / CONFIRMED",
            score,
            "Breadth and low correlation "
            "confirmed, RSP is starting "
            "to confirm the market broadening."
        )

    # --------------------------------------------------------
    # Broadening
    # --------------------------------------------------------

    if (
        correlation["average"] < 0.20
        and breadth["breadth200"] >= 68
    ):

        return (
            "🟡 BROADENING WATCH",
            score,
            "Correlation is very low and breadth "
            "is confirmed, but RSP has not yet confirmed "
            "relative strength."
        )

    # --------------------------------------------------------
    # Constructive
    # --------------------------------------------------------

    if (
        breadth["breadth200"] >= 65
        and rsp["above_ma20"]
    ):

        return (
            "🟡 CONSTRUCTIVE",
            score,
            "Market breadth is improving and "
            "RSP is starting to confirm the move."
        )

    # --------------------------------------------------------
    # Weak
    # --------------------------------------------------------

    if (
        breadth["breadth200"] < 55
        and not rsp["above_ma50"]
    ):

        return (
            "🔴 HYPOTHESIS BROKEN",
            score,
            "Breadth is weak and RSP is "
            "below its 50DMA."
        )

    return (
        "🟡 WATCH",
        score,
        "Market structure is mixed."
    )


# ============================================================
# REPORT
# ============================================================

def build_report(
    date,
    rsp,
    breadth,
    correlation,
    signal,
    score,
    explanation,
):

    lines = []

    lines.append(
        "📊 RSP BREADTH HYPOTHESIS MONITOR"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"Market date: {date.date()}"
    )

    lines.append("")

    # --------------------------------------------------------
    # RSP
    # --------------------------------------------------------

    lines.append(
        "1️⃣ RSP vs SPY"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"RSP/SPY: {rsp['ratio']:.4f}"
    )

    lines.append(
        f"Change: "
        f"{rsp['daily_change']:+.2f}%"
    )

    diff20 = (
        rsp["ratio"]
        / rsp["ma20"]
        - 1
    ) * 100

    diff50 = (
        rsp["ratio"]
        / rsp["ma50"]
        - 1
    ) * 100

    lines.append(
        f"20DMA: {rsp['ma20']:.4f} "
        f"({'✓' if rsp['above_ma20'] else '✗'})"
    )

    lines.append(
        f"50DMA: {rsp['ma50']:.4f} "
        f"({'✓' if rsp['above_ma50'] else '✗'})"
    )

    lines.append(
        f"vs 20DMA: {diff20:+.2f}%"
    )

    lines.append(
        f"vs 50DMA: {diff50:+.2f}%"
    )

    lines.append(
        f"RSP regime: "
        f"{classify_rsp(rsp)}"
    )

    lines.append(
        f"20D relative high: "
        f"{'YES' if rsp['relative_high_20'] else 'NO'}"
    )

    lines.append(
        f"63D relative high: "
        f"{'YES' if rsp['relative_high_63'] else 'NO'}"
    )

    lines.append(
        f"252D relative high: "
        f"{'YES' if rsp['relative_high_252'] else 'NO'}"
    )

    lines.append("")

    # --------------------------------------------------------
    # BREADTH
    # --------------------------------------------------------

    lines.append(
        "2️⃣ S&P 500 BREADTH"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"> 50DMA: "
        f"{breadth['breadth50']:.1f}% "
        f"({breadth['stocks50']}/"
        f"{breadth['valid50']})"
    )

    lines.append(
        f"> 200DMA: "
        f"{breadth['breadth200']:.1f}% "
        f"({breadth['stocks200']}/"
        f"{breadth['valid200']})"
    )

    lines.append(
        f"Breadth regime: "
        f"{breadth['regime']}"
    )

    lines.append("")

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    lines.append(
        "3️⃣ 3-MONTH INTRA-STOCK CORRELATION"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"Average: "
        f"{correlation['average']:.3f}"
    )

    lines.append(
        f"Median:  "
        f"{correlation['median']:.3f}"
    )

    lines.append(
        f"Regime: "
        f"{correlation['regime']}"
    )

    lines.append(
        f"Stocks: "
        f"{correlation['stocks']}"
    )

    lines.append(
        f"Pairs: "
        f"{correlation['pairs']:,}"
    )

    lines.append("")

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    lines.append(
        "4️⃣ HYPOTHESIS CONDITIONS"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        "Correlation < 0.200: "
        f"{'✓' if correlation['average'] < 0.20 else '✗'}"
    )

    lines.append(
        "Breadth ≥ 68.0%: "
        f"{'✓' if breadth['breadth200'] >= 68 else '✗'}"
    )

    lines.append(
        "RSP > 20DMA: "
        f"{'✓' if rsp['above_ma20'] else '✗'}"
    )

    lines.append(
        "20DMA > 50DMA: "
        f"{'✓' if rsp['ma20_above_ma50'] else '✗'}"
    )

    lines.append("")

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    lines.append(
        "5️⃣ SIGNAL"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        signal
    )

    lines.append(
        f"Score: {score}/4"
    )

    lines.append(
        explanation
    )

    lines.append("")

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    lines.append(
        "6️⃣ DYNAMIC MARKET STATE"
    )

    lines.append(
        "──────────────────────────────"
    )

    lines.append(
        f"Correlation: "
        f"{correlation['regime']}"
    )

    lines.append(
        f"Breadth: "
        f"{breadth['regime']}"
    )

    lines.append(
        f"RSP relative strength: "
        f"{classify_rsp(rsp)}"
    )

    lines.append("")

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    lines.append(
        "7️⃣ INTERPRETATION"
    )

    lines.append(
        "──────────────────────────────"
    )

    if correlation["average"] < 0.20:

        lines.append(
            "✓ Very low correlation: "
            "stocks are moving less in sync."
        )

    else:

        lines.append(
            "✗ Correlation is not low."
        )

    if breadth["breadth200"] >= 68:

        lines.append(
            "✓ Breadth CONFIRMED: "
            "market breadth is "
            "in the confirmation zone."
        )

    elif breadth["breadth200"] >= 60:

        lines.append(
            "• Breadth is recovering."
        )

    else:

        lines.append(
            "✗ Breadth is weak."
        )

    if rsp["above_ma20"]:

        lines.append(
            "✓ RSP above 20DMA."
        )

    elif rsp["above_ma50"]:

        lines.append(
            "• RSP above 50DMA, "
            "but still below 20DMA."
        )

    else:

        lines.append(
            "✗ RSP relative strength "
            "not confirmed."
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def run():

    args = parse_args()

    initialize_database()

    symbols = get_sp500_symbols()

    close = download_incremental(
        symbols,
        args,
    )

    date = select_date(
        close,
        args.date,
    )

    print(
        f"\nUsing market date: "
        f"{date.date()}"
    )

    rsp = calculate_rsp_spy(
        close,
        date,
    )

    breadth = calculate_breadth(
        close,
        date,
    )

    correlation = calculate_correlation(
        close,
        date,
    )

    signal, score, explanation = (
        generate_signal(
            breadth,
            correlation,
            rsp,
        )
    )

    report = build_report(
        date,
        rsp,
        breadth,
        correlation,
        signal,
        score,
        explanation,
    )

    print()
    print(report)
    print()

    if args.save:

        timestamp = datetime.now(
            TZ
        ).isoformat()

        result = {

            "timestamp":
                timestamp,

            "market_date":
                date.date().isoformat(),

            "rsp_spy":
                rsp["ratio"],

            "rsp_spy_daily_change":
                rsp["daily_change"],

            "rsp_spy_20dma":
                rsp["ma20"],

            "rsp_spy_50dma":
                rsp["ma50"],

            "rsp_above_20dma":
                int(rsp["above_ma20"]),

            "rsp_above_50dma":
                int(rsp["above_ma50"]),

            "rsp_20dma_above_50dma":
                int(rsp["ma20_above_ma50"]),

            "rsp_relative_high_20d":
                int(rsp["relative_high_20"]),

            "rsp_relative_high_63d":
                int(rsp["relative_high_63"]),

            "rsp_relative_high_252d":
                int(rsp["relative_high_252"]),

            "breadth_50dma":
                breadth["breadth50"],

            "breadth_200dma":
                breadth["breadth200"],

            "stocks_above_50dma":
                breadth["stocks50"],

            "stocks_above_200dma":
                breadth["stocks200"],

            "correlation_average":
                correlation["average"],

            "correlation_median":
                correlation["median"],

            "correlation_stocks":
                correlation["stocks"],

            "correlation_pairs":
                correlation["pairs"],

            "breadth_regime":
                breadth["regime"],

            "correlation_regime":
                correlation["regime"],

            "rsp_regime":
                classify_rsp(rsp),

            "signal":
                signal,

            "score":
                score,
        }

        save_analysis(
            result
        )

        print(
            "✓ Analysis saved to SQLite."
        )

    if args.telegram:

        send_message(
            report
        )


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