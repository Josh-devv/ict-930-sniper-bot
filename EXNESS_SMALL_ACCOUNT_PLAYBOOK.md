# 🚀 Exness Small Account Playbook ($20 – $100)
## ICT 9:30 Sniper Strategy on USTEC / NAS100

---

## 1. Overview & Math

On retail CFD brokers like **Exness**, you don't need $1,000+ margin to trade the Nasdaq-100. By using **0.01 micro lots** on `USTEC`, your risk per trade is reduced to pocket change while keeping the exact same 4.9R mathematical edge.

### Risk & Profit Math on Exness USTEC (0.01 Lot)
- **1 point on 0.01 lot** $\approx$ **$0.10 USD**
- **Average Stop Loss (15 points)** = **$1.50 USD**
- **Average Target (50 points)** = **+$5.00 USD**
- **Risk-to-Reward Ratio**: **~3.3R to 5.0R**

---

## 2. Compounding Schedule ($50 to $1,000+)

| Account Tier | Account Balance | Lot Size | Risk per Trade (15pt SL) | Profit per Win (50pt TP) | Target Balance to Level Up |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Level 1** | **$20 – $50** | **0.01 lot** | **$1.50** | **+$5.00** | Reach $100 |
| **Level 2** | **$100 – $200** | **0.02 lot** | **$3.00** | **+$10.00** | Reach $250 |
| **Level 3** | **$250 – $500** | **0.05 lot** | **$7.50** | **+$25.00** | Reach $600 |
| **Level 4** | **$600 – $1,200**| **0.10 lot** | **$15.00** | **+$50.00** | Reach $1,500 |

> **Discipline Rule**: Never increase lot size after a losing trade. Only level up when your balance permanently crosses the next tier threshold.

---

## 3. How to Set Up Instant Phone Alerts (TradingView App)

1. Open your **TradingView mobile app** on your phone (log into your account).
2. On your computer chart (with `TradingView_Indicator_Script.pine` loaded):
   - Right-click chart $\rightarrow$ **Add Alert**.
   - Condition: `ICT 9:30 Sniper — VWAP Band Reversal v6`.
   - Select `🔴 SHORT Signal Triggered` or `🟢 LONG Signal Triggered`.
   - Under **Notifications**: Check **"Notify in App"** and **"Play Sound"**.
3. At 9:30 AM EST:
   - Your phone buzzes with the exact entry and stop level.
   - Open your Exness app $\rightarrow$ Select `USTEC` $\rightarrow$ Enter 0.01 lot with the given SL/TP.

---

## 4. Automated Python Script for Exness MT5

The Python script [`scripts/bot_exness_mt5.py`](file:///c:/Users/JOSHUA/Desktop/10am/scripts/bot_exness_mt5.py) connects directly to your Exness MetaTrader 5 terminal:
- Monitors live 9:30 AM open on USTEC.
- Detects the 2.0σ VWAP band rejection.
- Automatically places the 0.01 lot order with Stop Loss and Take Profit directly in your Exness account!

