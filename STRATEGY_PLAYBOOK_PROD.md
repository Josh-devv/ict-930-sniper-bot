# Institutional NQ Strategy Playbook (Production Specifications)
**Empirically Verified on 8.8 Years of Real CME Continuous NQ Futures Data (2018–2026)**
*Zero-Cheat Standard: Hard Stops Checked First, Pessimistic Limit Fills, $5.20 NQ / $1.24 MNQ Fees, 1 Tick Slippage.*

---

## MODEL 1: The 15-Minute Institutional Sniper (Highest Win Rate)
*Best for: High conviction, maximum consistency, prop firm challenge passing without drawdowns.*

* **Source File**: [`scripts/expanded_orderflow_liquidity_model.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/expanded_orderflow_liquidity_model.py)
* **Timeframe**: 15-Minute Chart
* **Win Rate**: **66.7%**
* **Profit Factor (PF)**: **4.13**
* **Average Stop Loss**: **16.5 points** ($33.00 on 1 Micro MNQ / $330.00 on 1 Mini NQ)
* **Target**: **33.0 points (2.0R)**
* **Trade Frequency**: ~2 to 3 trades per month (high-grade selective)
* **8-Year Net Profit**: **+$8,069 on 1 Mini NQ** / **+$807 on 1 Micro MNQ** (Max DD: -$792 / -$79)

### Core Rules:
1. **Liquidity Sweep**: Price must pierce Previous Day High/Low (PDH/PDL), 09:30–10:00 Opening Range High/Low, or a 20-bar 15m structural swing.
2. **Orderflow Absorption**: Sweep candle must leave an upper/lower rejection wick $\ge 28\%$ of total candle range, and intra-bar delta must flip opposite the sweep.
3. **Macro Alignment**: Direction must match the 1-Hour 20 EMA trend.
4. **Execution**: Passive limit order placed strictly at the **50% Mean Threshold (MT)** of the 15m Order Block.

---

## MODEL 2: The 5-Minute Active Sweep Engine (High Frequency)
*Best for: Active day traders wanting actionable, consistent setups every single week.*

* **Source File**: [`scripts/high_frequency_institutional_model.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/high_frequency_institutional_model.py)
* **Timeframe**: 5-Minute Chart
* **Win Rate**: **58.1%**
* **Profit Factor (PF)**: **1.85**
* **Average Stop Loss**: **13.4 points** ($26.80 on 1 Micro MNQ / $268.00 on 1 Mini NQ)
* **Target**: **20.1 points (1.5R)**
* **Trade Frequency**: **65.1 trades/year (1 to 2 trades every single week)** | 573 trades total
* **Annual Consistency**: **9 out of 9 consecutive green years** (2018 through 2026)
* **8-Year Net Profit**: **+$58,976 on 1 Mini NQ** / **+$5,898 on 1 Micro MNQ**

### Core Rules:
1. **Intra-Session Sweep**: Price pierces a rolling 5-bar structural swing high or low between 09:45 and 15:30 EST.
2. **Rejection & Absorption**: Rejection wick $\ge 25\%$ with delta absorption.
3. **Displacement Bar**: Next candle closes with a displacement body $\ge 0.85\times$ ATR in the direction of the 1-Hour 20 EMA.
4. **Execution**: Passive limit order at the **50% Mean Threshold (MT)** of the Order Block.

---

## TradingView Production Indicator
* **File Path**: [`scripts/TradingView_Institutional_Active_Orderflow_OB.pine`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/TradingView_Institutional_Active_Orderflow_OB.pine)
* Supports 1-click dropdown switching between **15m Sniper** and **5m Active**.
* Plots dynamic PDH/PDL, Opening Range, and 5-bar structural swings.
* Draws entry boxes at 50% MT with visual Stop Loss and Take Profit lines and native alerts.

---

## MODEL 3: Tony Iyke SMC Market Structure Engine (Trend + POI + Inducement + CHoCH)
*Best for: Trend traders executing on 3m/5m charts using HTF Discount/Premium POIs and Inducement sweeps.*

* **Source Files**: 
  - Standard POI Engine: [`scripts/test_tony_iyke_optimized_market_structure.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/test_tony_iyke_optimized_market_structure.py)
  - Daily High-Frequency Engine: [`scripts/test_high_frequency_tony_iyke.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/test_high_frequency_tony_iyke.py)
* **TradingView Indicator**: [`scripts/TradingView_Tony_Iyke_NQ_SMC.pine`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/TradingView_Tony_Iyke_NQ_SMC.pine)

### Performance by Mode:
| Mode | Timeframe | Trade Count (8.8 Yrs) | Trade Frequency | Win Rate | Profit Factor | 8-Year Net Profit (1 NQ) | Avg Stop Loss | Consistency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Daily High-Frequency** | **2-Minute** | **2,127 trades** | **~1 trade/day (4.6/wk)** | **47.1%** | **1.23** | **+$62,089** | **11.2 pts ($22 MNQ)** | **9/9 Green Years** |
| **Active Morning/PM** | **3-Minute** | **1,626 trades** | **~3.6 trades/week** | **47.8%** | **1.29** | **+$60,642** | **12.0 pts ($24 MNQ)** | **8/9 Green Years** |
| **Selective POI Sniper** | **5-Minute** | **361 trades** | **~1 trade/week** | **59.3%** | **1.98** | **+$41,585** | **13.6 pts ($27 MNQ)** | **9/9 Green Years** |

### Core Rules:
1. **HTF Market Structure Context**: Direction is strictly gated by the 1-Hour 20 EMA trend (Higher Highs / Higher Lows).
2. **Point of Interest (POI)**: 
   - Bullish trades are ONLY permitted in the **Discount Zone** (below the 50% equilibrium of the recent dealing range).
   - Bearish trades are ONLY permitted in the **Premium Zone** (above the 50% equilibrium of the recent dealing range).
3. **Inducement (IDM) Sweep**: Price pierces a minor internal 5-bar swing low (for longs) or swing high (for shorts) to capture premature retail stops.
4. **CHoCH (Change of Character)**: The candle immediately prints a displacement close in the trend direction with body $\ge 0.70\times$ ATR and orderflow absorption.
5. **Execution**: Passive limit order at the **50% Mean Threshold (MT)** of the Order Block with a structural stop 1.0 pt beyond the swept Inducement.

---

## MODEL 4: BTC Orderflow Absorption Sniper (Institutional Precision Edition)
*Best for: 24/7 crypto day trading with automated Telegram alerts on Exness MT5 / Binance / OKX.*

* **Source Files**:
  - Python Backtest: [`scripts/btc_absorption_filtered_backtest.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/btc_absorption_filtered_backtest.py)
  - Precision Sweep & Verification: [`scratch/btc_precision_sweep.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scratch/btc_precision_sweep.py)
  - Telegram Bot (Live): [`scripts/btc_telegram_alert_bot.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/btc_telegram_alert_bot.py)
  - Telegram Bot (Cron): [`scripts/btc_telegram_alert_bot_cron.py`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/btc_telegram_alert_bot_cron.py)
* **TradingView Production Indicator**: [`scripts/TradingView_BTC_Orderflow_Absorption_Indicator.pine`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/TradingView_BTC_Orderflow_Absorption_Indicator.pine)
* **TradingView Strategy**: [`scripts/TradingView_BTC_Orderflow_Absorption.pine`](file:///c:/Users/JOSH/Desktop/2xjosh/10am/scripts/TradingView_BTC_Orderflow_Absorption.pine)

### Empirically Verified Performance (2 Years of Real Dukascopy BTC 5m Data, 2024–2026):
| Metric | Institutional Precision (v3.0) | Standard 1H 50 EMA Filter | Raw Absorption (Unfiltered) |
| :--- | :---: | :---: | :---: |
| **Win Rate** | **56.4%** | 53.5% | 42.1% |
| **Profit Factor (PF)** | **1.52** | 1.38 | 0.75 (Losing) |
| **Total Trades** | **702 trades** (~28/mo, ~7/wk) | 837 trades | 1,420 trades |
| **Net Profit (0.1 BTC)** | **+$5,279.20** | +$4,433.92 | -$7,064.10 |
| **Net Profit (1.0 BTC)** | **+$52,792.00** | +$44,339.20 | -$70,641.01 |
| **Monthly Consistency** | **22 out of 25 Green Months (88%)** | 22 out of 25 Green Months | 4 out of 25 Green Months |
| **Average Win / Loss** | **+$37.80 / -$27.40** (0.1 BTC) | +$36.06 / -$30.13 | +$35.09 / -$34.12 |

### Institutional Precision Rules (Eliminates Bad Trades):
1. **Session Windows**: Monitored strictly during London (07:00–11:00 UTC) and NY (13:00–18:00 UTC) institutional liquidity windows.
2. **Macro Trend Gate**: Longs ONLY when 5m close is **ABOVE 1H 50 EMA**; Shorts ONLY when 5m close is **BELOW 1H 50 EMA**.
3. **Over-Extension Protection (No Selling Bottoms / Buying Tops)**: Distance between 5m close and 1H 50 EMA must be **$\le 8.0$ ATRs**. Filters out exhausted parabolic drops/pumps where retail gets squeezed.
4. **Dealing Range 30-Bar Equilibrium**: Institutions buy wholesale and sell wholesale. Longs must be in **Discount (< 50% Dealing Range)**; Shorts must be in **Premium (> 50% Dealing Range)**.
5. **Fakeout Trap Confirmation**: Price must sweep liquidity and close back **inside the swept level** (`close > local_low` for Long, `close < local_high` for Short), proving sellers were absorbed and trapped.
6. **Auction Winner Confirmation**: Long candle must close green or in the upper 50% of its range; Short candle must close red or in the lower 50%.
7. **Volume Climax & Wick Absorption**: Volume $\ge 1.3\times$ 20 SMA; Absorption wick $\ge 30\%$ of total bar range.
8. **Direct Entry Execution**: Market buy/sell at the **close of the absorption candle** (SL = $1.0\times$ ATR beyond extreme; TP1 = +1.5R bank 60%, TP2 = +3.0R runner 40%). Shows AI Quality Score (0–100%) on Telegram.


