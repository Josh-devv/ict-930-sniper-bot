# 🏆 Prop Firm Consistency Rules & Pass Guide
## TopStep ($50k) & FundedNext ($50k Flex / Futures)

---

## 1. TopStep $50,000 Trading Combine

| Parameter | TopStep $50k Requirement | How the 9:30 Sniper Bot Complies |
|---|---|---|
| **Profit Target** | **+$3,000** | Passes in ~2 to 2.5 weeks |
| **Max Trailing Drawdown** | **-$2,000** *(calculated End-of-Day)* | Bot worst observed DD is **-$1,251** (100% safe) |
| **Daily Loss Limit** | **No hard rule** *(soft lock at $1,000)* | Bot stops at -$800 via Prop Firm Guardian |
| **Contract Sizing** | Up to 5 contracts allowed | Bot trades strictly **1 contract** (safe & disciplined) |
| **Minimum Trading Days** | **None** *(can pass as soon as target + consistency are met)* | Takes ~6 to 10 trading days naturally |

### 🎯 TopStep Consistency Rule: The 50% Rule
- **The Rule**: No single trading day can account for **50% or more** of your total profit upon passing.
- **The Calculation**:
  $$\text{Target} = \$3,000 \implies \text{Max single day allowed} = \$1,500 \text{ (50\%)}$$
- **What to do if you have a huge win?**
  - If the bot makes **+$1,600** on Day 1, your account is **NOT failed!**
  - TopStep simply requires you to keep trading until your total balance reaches **$3,200+** (so that $1,600 is $\le 50\%$ of your total profit).
- **Bot Behavior**: Average win is **+$950**, meaning it naturally hits the target across **3 to 4 separate winning days**, easily staying under 50% per day.

---

## 2. FundedNext $50,000 Challenge (Flex / Futures)

| Parameter | FundedNext $50k Requirement | How the 9:30 Sniper Bot Complies |
|---|---|---|
| **Profit Target** | **+$2,500** (5%) or **+$3,000** (6%) | Passes in ~1.8 to 2.5 weeks |
| **Overall Max Drawdown** | **-$2,000** (4%) on Futures | Bot worst historical DD is **-$1,251** (safe) |
| **Max Daily Loss** | **-$1,000** on Futures | Average loss is only **-$260** (safe) |
| **Minimum Trading Days** | **5 Minimum Days** | Bot trades ~3.5 days/week (takes ~10 days) |

### 🎯 FundedNext Consistency Pillars

#### A. The Best-Day Profit Consistency Rule (30% – 40% Rule)
- **The Rule**: On consistency-enforced plans (Flex/Stellar), no single trading day can make up more than **40% of your total profit**.
- **The Calculation**:
  $$\text{Target} = \$2,500 \implies \text{Max profit in one day} = \$1,000 \text{ (40\%)}$$
- **How to manage this**:
  - If the bot hits a monster target and makes **+$1,200** in one day, you do **not** fail! You simply let the bot run for another 2–3 days to increase the total profit to **+$3,000+** (so that $1,200 is 40% of $3,000).

#### B. Lot Size / Contract Consistency Rule
- **The Rule**: Your position size must stay within **0.5x to 2.0x of your average size**. You cannot gamble 5 contracts on Monday and 0.1 on Tuesday.
- **Bot Compliance**: **100% Perfect.** The bot is hard-coded to trade **strictly 1 contract** on every execution. Your lot-size consistency score will be **100% flawless**.

#### C. Order / Strategy Consistency
- **The Rule**: No martingale, no grid hedging, no holding over weekends.
- **Bot Compliance**: The bot enters with an institutional bracket order (Entry, SL, TP) and **flattens at 15:45 EST every day** (never holds overnight or over weekends).

---

## 3. Step-by-Step Passing Checklist

1. **Keep Contract Sizing Fixed**: Ensure `CONTRACTS=1` in your `.env` settings.
2. **1 Trade Per Day Rule**: Once the daily setup triggers and exits, never take manual trades.
3. **If You Have a Big Day**: If Day 1 makes $1,200+, do not stop trading immediately. Let the bot run for another 2–3 days so the profit is evenly distributed across 4–5 days.
4. **Minimum 5 Days**: FundedNext requires trading on at least 5 separate days before requesting the funded certificate. The bot naturally takes ~6–10 days to hit the target.
