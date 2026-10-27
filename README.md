# market-regime-monitor

A small Dockerised service that every US trading day at **10:00 New York time** reads the market's
"weather" — breadth, risk-on/risk-off, macro stress — and posts plain-language reports to Telegram.

It is deliberately **not a price forecaster**. Each monitor answers a narrower question: *do the
different parts of the market agree with each other?* Divergence between them is the thing worth
watching. Whether such signals actually lead regime changes was tested separately, with mixed results —
see [regime-watch](https://github.com/pnnmail/regime-watch).

## The four monitors

| Monitor | Question | Inputs |
|---|---|---|
| **RSP Breadth** (`theories/rsp_breadth.py`) | Is the rally broad or carried by a few mega-caps? | Equal-weight vs cap-weight S&P 500 (RSP/SPY) vs its 20/50-DMA; % of S&P 500 stocks above 50/200-DMA; 3-month average pairwise correlation across ~500 stocks (125k pairs) |
| **Duration / Volatility** (`theories/duration_volatility.py`) | Risk-on, risk-off, or are bonds and stocks telling different stories? | TLT/IEF/SHY, VIX, SPY trend, HYG/LQD credit, 10Y yield, dollar |
| **Macro Stress** (`theories/macro_stress.py`) | Is the bond market pricing a slowdown that equities ignore? | Long vs short duration, 10Y yield, VIX, SPY vs 50/200-DMA, HYG/LQD |
| **BTC Momentum** (`theories/daily_strategy_check.py`) | Time-series momentum with volatility scaling on BTC; keeps a journal and scores its own accuracy | BTC daily closes |

Regimes reported by the duration/macro monitors:

```
NORMAL RISK-ON      SPY ↑ + credit healthy + VIX low + TLT neutral
EARLY WARNING       TLT starts outperforming IEF/SHY, equities and credit still strong
MACRO DISLOCATION   TLT ↑ + 10Y yield ↓ + SPY strong + VIX low + credit healthy   ← main observation zone
MACRO STRESS        Treasuries strong + equities weakening + credit deteriorating
RISK-OFF            Treasuries strong + equities weak + credit weak + VIX high
```

## Example output (2026-10-27)

Full reports are in [`examples/`](examples/). Excerpt from the breadth monitor:

```
1️⃣ RSP vs SPY            RSP/SPY 0.2820, above 20DMA and 50DMA  → 🟢 IMPROVING
2️⃣ S&P 500 BREADTH       > 50DMA: 58.8%   > 200DMA: 57.2%      → 🟡 RECOVERY
3️⃣ INTRA-STOCK CORR.     average 0.109 across 125,751 pairs    → 🟢 LOW
4️⃣ CONDITIONS            correlation < 0.20 ✓   breadth ≥ 68% ✗   RSP > 20DMA ✓
```

The duration and macro monitors both read **MIXED** that day: no confirmed risk-on, no confirmed risk-off.

## Track record — the honest part

- **BTC momentum signal:** 42 daily signals logged since late August 2026. Of the 35 checked at the
  7-day horizon, **14 right and 21 wrong (40% accuracy)**; mean 7-day return after a LONG signal
  +0.91%. Not better than a coin flip, so the daily BTC Telegram message is now off by default
  (`BTC_TELEGRAM=0`). The journal keeps updating (`results/btc_strategy_history.json`).
- **Breadth, VIX term structure and credit as leading indicators** were tested in
  [regime-watch](https://github.com/pnnmail/regime-watch) on 2015–2026 data. Breadth *coincides* with
  regime changes rather than leading them; the VIX term structure is significant in-sample but adds
  nothing to a standard volatility forecast out of sample. So these monitors are a dashboard that
  describes the present, not a timing tool.

## Engineering notes

- Telegram messages are split at 3,900 characters (one report reached 4,328 and got HTTP 400), and the
  bot token is stripped from any error text before logging.
- The S&P 500 constituent list is read from Wikipedia via `StringIO` (newer pandas treats a raw HTML
  string as a file path and silently fell back to a stale cached list).
- Prices for ~500 stocks are cached in SQLite and updated incrementally.
- `theories/` is a bind mount, so edits apply on the next run without rebuilding the image.

## Run

```bash
cp .env.example .env          # Telegram token and chat id
docker compose up -d --build  # runs Mon–Fri at 10:00 America/New_York
```

Or run a single monitor locally:

```bash
pip install -r requirements.txt
python -m theories.duration_volatility          # print report
python -m theories.macro_stress --save --telegram
```
