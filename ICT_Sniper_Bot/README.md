# ICT 9:30 VWAP Sniper Bot
## Professional Automated Trading Bot for NQ Futures

---

## What This Bot Does

This bot automatically trades the **ICT 9:30 Manipulation + VWAP Reversal** strategy on NQ (Nasdaq-100) futures — compatible with **FundedNext** and **TopStep** funded accounts.

### Backtest Results (49 Real Trading Days)
| Metric | Result |
|---|---|
| Win Rate | **58%** |
| Avg R:R | **4.9R** |
| Net Profit (1 NQ) | **+$13,771** |
| Max Drawdown | **-$565** |
| Trades per week | ~3 |

---

## Requirements

- Windows 10 or 11 PC (or any VPS)
- Python 3.10+ installed ([python.org](https://python.org))
- A Tradovate account (FundedNext / TopStep both use Tradovate)
- Tradovate API credentials (free — see Step 2 below)

---

## Setup Instructions (takes ~5 minutes)

### Step 1 — Install Python
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download the latest version
3. **IMPORTANT:** Check the box **"Add Python to PATH"** during install

### Step 2 — Get Tradovate API Credentials
1. Log in at [trader.tradovate.com](https://trader.tradovate.com)
2. Click your name → **Settings** → **API Access**
3. Click **Create Application** → name it anything (e.g. "Sniper Bot")
4. Copy your **CID** and **Secret** — you'll need these in Step 3

### Step 3 — Install & Configure the Bot
1. Double-click **`1_INSTALL_AND_SETUP.bat`**
2. Follow the on-screen prompts — enter your Tradovate email, password, CID, and Secret
3. When asked "Live trading?" — type **NO** first (run demo first!)

### Step 4 — Run the Bot
1. Double-click **`2_START_BOT.bat`**
2. The bot will connect to Tradovate and wait for market open (9:25 AM ET)
3. All activity is logged in the **`logs/`** folder

---

## How the Strategy Works

```
Every day at 9:30 AM — market opens
Bot watches for price to spike into VWAP deviation band
If candle shows a rejection wick (≥35% of range)
    → Limit order placed at 50% of the wick (CE point)
    → Stop Loss set 2pts beyond the wick tip
    → Take Profit set at the session VWAP line
Only 1 trade per day maximum
Bot goes flat at 3:45 PM automatically
```

---

## Important Rules

- Always run in **DEMO mode** for at least 1 week before going live
- The bot trades **1 contract only** — do not change this during a challenge
- **Do not close the bot window** while a trade is active
- Check the `logs/` folder daily to see what the bot is doing
- Update `NQ_SYMBOL` in the `.env` file when the futures contract rolls over (every 3 months)

---

## Going Live

When you are ready to trade real money:
1. Open the `.env` file (in the bot folder) with Notepad
2. Change `LIVE_MODE=false` to `LIVE_MODE=true`
3. Save the file
4. Restart the bot

---

## Current NQ Contract Symbols
| Month | Symbol |
|---|---|
| December 2026 | NQZ6 |
| March 2027 | NQH7 |
| June 2027 | NQM7 |

Update `NQ_SYMBOL=` in your `.env` file when the contract changes (check Tradovate for the active contract).

---

## Support

Check `logs/bot_log.txt` for all bot activity. Every signal, order, and exit is logged with timestamps.

