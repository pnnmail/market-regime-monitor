import os
import sqlite3
from datetime import datetime


DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "market_lab.db")


def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_FILE)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    return conn


def initialize_database():

    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_prices (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            close REAL,
            PRIMARY KEY (date, symbol)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_prices_date
        ON market_prices(date)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_prices_symbol
        ON market_prices(symbol)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,
            market_date TEXT NOT NULL,

            rsp_spy REAL,
            rsp_spy_daily_change REAL,
            rsp_spy_20dma REAL,
            rsp_spy_50dma REAL,

            rsp_above_20dma INTEGER,
            rsp_above_50dma INTEGER,
            rsp_20dma_above_50dma INTEGER,

            rsp_relative_high_20d INTEGER,
            rsp_relative_high_63d INTEGER,
            rsp_relative_high_252d INTEGER,

            breadth_50dma REAL,
            breadth_200dma REAL,

            stocks_above_50dma INTEGER,
            stocks_above_200dma INTEGER,

            correlation_average REAL,
            correlation_median REAL,

            correlation_stocks INTEGER,
            correlation_pairs INTEGER,

            breadth_regime TEXT,
            correlation_regime TEXT,
            rsp_regime TEXT,

            signal TEXT,
            score INTEGER,

            UNIQUE(market_date)
        )
        """
    )

    conn.commit()
    conn.close()


def get_last_market_date():

    initialize_database()

    conn = get_connection()

    row = conn.execute(
        """
        SELECT MAX(date)
        FROM market_prices
        """
    ).fetchone()

    conn.close()

    if not row or row[0] is None:
        return None

    return row[0]


def get_database_stats():

    initialize_database()

    conn = get_connection()

    dates = conn.execute(
        """
        SELECT COUNT(DISTINCT date)
        FROM market_prices
        """
    ).fetchone()[0]

    symbols = conn.execute(
        """
        SELECT COUNT(DISTINCT symbol)
        FROM market_prices
        """
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT COUNT(*)
        FROM market_prices
        """
    ).fetchone()[0]

    first_date = conn.execute(
        """
        SELECT MIN(date)
        FROM market_prices
        """
    ).fetchone()[0]

    last_date = conn.execute(
        """
        SELECT MAX(date)
        FROM market_prices
        """
    ).fetchone()[0]

    conn.close()

    return {
        "dates": dates,
        "symbols": symbols,
        "rows": rows,
        "first_date": first_date,
        "last_date": last_date,
    }


def save_prices(close):

    if close is None or close.empty:
        return

    initialize_database()

    conn = get_connection()

    rows = []

    for date, row in close.iterrows():

        date_str = date.strftime("%Y-%m-%d")

        for symbol, value in row.items():

            if value is None:
                continue

            try:
                value = float(value)
            except Exception:
                continue

            if value != value:
                continue

            rows.append(
                (
                    date_str,
                    str(symbol),
                    value,
                )
            )

    if rows:

        conn.executemany(
            """
            INSERT OR REPLACE INTO market_prices
            (date, symbol, close)
            VALUES (?, ?, ?)
            """,
            rows,
        )

        conn.commit()

    conn.close()


def load_prices():

    initialize_database()

    conn = get_connection()

    import pandas as pd

    df = pd.read_sql_query(
        """
        SELECT date, symbol, close
        FROM market_prices
        ORDER BY date
        """,
        conn,
    )

    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])

    matrix = df.pivot(
        index="date",
        columns="symbol",
        values="close",
    )

    matrix = matrix.sort_index()

    return matrix


def save_analysis(result):

    initialize_database()

    conn = get_connection()

    conn.execute(
        """
        INSERT OR REPLACE INTO analysis_results (

            timestamp,
            market_date,

            rsp_spy,
            rsp_spy_daily_change,
            rsp_spy_20dma,
            rsp_spy_50dma,

            rsp_above_20dma,
            rsp_above_50dma,
            rsp_20dma_above_50dma,

            rsp_relative_high_20d,
            rsp_relative_high_63d,
            rsp_relative_high_252d,

            breadth_50dma,
            breadth_200dma,

            stocks_above_50dma,
            stocks_above_200dma,

            correlation_average,
            correlation_median,

            correlation_stocks,
            correlation_pairs,

            breadth_regime,
            correlation_regime,
            rsp_regime,

            signal,
            score

        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            result["timestamp"],
            result["market_date"],

            result["rsp_spy"],
            result["rsp_spy_daily_change"],
            result["rsp_spy_20dma"],
            result["rsp_spy_50dma"],

            result["rsp_above_20dma"],
            result["rsp_above_50dma"],
            result["rsp_20dma_above_50dma"],

            result["rsp_relative_high_20d"],
            result["rsp_relative_high_63d"],
            result["rsp_relative_high_252d"],

            result["breadth_50dma"],
            result["breadth_200dma"],

            result["stocks_above_50dma"],
            result["stocks_above_200dma"],

            result["correlation_average"],
            result["correlation_median"],

            result["correlation_stocks"],
            result["correlation_pairs"],

            result["breadth_regime"],
            result["correlation_regime"],
            result["rsp_regime"],

            result["signal"],
            result["score"],
        ),
    )

    conn.commit()
    conn.close()


def get_analysis_history(limit=100):

    initialize_database()

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM analysis_results
        ORDER BY market_date DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    conn.close()

    return rows