//+------------------------------------------------------------------+
//|                                  BTC_Absorption_Sniper_EA.mq5    |
//|                 ICT Orderflow CVD Absorption Sniper Alert EA     |
//|                                    Copyright 2026, Antigravity   |
//+------------------------------------------------------------------+
#property copyright "Antigravity ICT Trading"
#property link      "https://github.com/Josh-devv/ict-930-sniper-bot"
#property version   "1.00"
#property strict

// ===================================================================
// INPUT PARAMETERS
// ===================================================================
input group "=== Strategy Parameters ==="
input double   InpVolumeMultiplier = 1.3;    // Volume Climax Multiplier (vs 20 SMA)
input double   InpWickRatio        = 0.30;   // Min Lower Wick Ratio (30%)
input int      InpLookback         = 20;     // Local Liquidity Sweep Lookback
input double   InpATRMultiplier    = 1.0;    // Stop Loss ATR Multiplier
input double   InpRR_TP1           = 1.5;    // TP1 Multiplier (Bank 60%)
input double   InpRR_TP2           = 3.0;    // TP2 Multiplier (Runner 40%)

input group "=== Session Windows (UTC) ==="
input bool     InpEnableLondon     = true;   // Enable London (07:00 - 11:00 UTC)
input bool     InpEnableNY         = true;   // Enable NY (13:00 - 18:00 UTC)

input group "=== Alert & Notification Options ==="
input bool     InpSendPushNotification = true; // Send Free Push to MT5 Mobile App
input bool     InpPlaySoundAlert       = true; // Play Alert Sound on Laptop
input bool     InpSendEmailAlert       = false;// Send Email Alert

// Global variables
datetime g_last_signal_time = 0;
int      g_atr_handle       = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== BTC Orderflow Absorption Sniper Alert EA Initialized ===");
   Print("Symbol: ", _Symbol, " | Timeframe: ", EnumToString((ENUM_TIMEFRAMES)_Period));

   // Create ATR handle
   g_atr_handle = iATR(_Symbol, _Period, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("Error creating ATR handle: ", GetLastError());
      return(INIT_FAILED);
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
}

//+------------------------------------------------------------------+
//| Helper: Check if Time is in Active Sessions (UTC)                |
//+------------------------------------------------------------------+
bool IsActiveSession(datetime dt)
{
   MqlDateTime mql_time;
   TimeToStruct(dt, mql_time);
   int hour = mql_time.hour;

   bool in_london = InpEnableLondon && (hour >= 7 && hour <= 11);
   bool in_ny     = InpEnableNY     && (hour >= 13 && hour <= 18);

   return (in_london || in_ny);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only evaluate on the OPEN of a new candle (Bar 1 just closed)
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == g_last_signal_time)
      return; // Already checked this bar

   // Bar 1 represents the newly closed 5m bar
   datetime bar1_time = iTime(_Symbol, _Period, 1);
   if(!IsActiveSession(bar1_time))
   {
      g_last_signal_time = current_bar_time;
      return;
   }

   // 1. Fetch OHLCV for Bar 1
   double open1  = iOpen(_Symbol, _Period, 1);
   double high1  = iHigh(_Symbol, _Period, 1);
   double low1   = iLow(_Symbol, _Period, 1);
   double close1 = iClose(_Symbol, _Period, 1);
   long   vol1   = iTickVolume(_Symbol, _Period, 1);

   // 2. Volume 20 SMA
   long sum_vol = 0;
   for(int i = 1; i <= 20; i++)
      sum_vol += iTickVolume(_Symbol, _Period, i);
   double vol_ma = (double)sum_vol / 20.0;
   bool vol_climax = (vol1 >= vol_ma * InpVolumeMultiplier);

   // 3. Local 20-period Low Sweep
   double local_low = DBL_MAX;
   for(int i = 2; i <= InpLookback + 1; i++)
   {
      double l = iLow(_Symbol, _Period, i);
      if(l < local_low)
         local_low = l;
   }
   bool low_swept = (low1 <= local_low);

   // 4. Candle Range & Lower Wick Calculation
   double range1 = MathMax(0.5, high1 - low1);
   double body_bottom = MathMin(open1, close1);
   double lower_wick = body_bottom - low1;
   double lower_wick_ratio = lower_wick / range1;

   bool wick_ok = (lower_wick_ratio >= InpWickRatio);

   // 5. Signal Trigger
   if(vol_climax && low_swept && wick_ok)
   {
      g_last_signal_time = current_bar_time;

      // Fetch ATR for stop buffer
      double atr_vals[1];
      if(CopyBuffer(g_atr_handle, 0, 1, 1, atr_vals) <= 0)
      {
         Print("Failed to get ATR value");
         return;
      }
      double atr = atr_vals[0];

      // Exact Execution Levels
      double entry_price = low1 + (0.50 * lower_wick); // 50% Wick Limit
      double stop_price  = low1 - (InpATRMultiplier * atr);
      double stop_dist   = entry_price - stop_price;

      if(stop_dist < 40.0) // Filter micro stops
         return;

      double tp1_price = entry_price + (InpRR_TP1 * stop_dist);
      double tp2_price = entry_price + (InpRR_TP2 * stop_dist);

      // Format Message
      string msg = StringFormat(
         "🚀 BTC ABSORPTION BUY SIGNAL!\n" +
         "Symbol: %s (5m)\n" +
         "-------------------------\n" +
         "🔹 BUY LIMIT (50%% Wick): %.2f\n" +
         "🔻 STOP LOSS: %.2f (Risk: $%.2f)\n" +
         "🟢 TP1 (60%% Close + BE): %.2f (+%.1fR)\n" +
         "🎯 TP2 (40%% Runner): %.2f (+%.1fR)",
         _Symbol, entry_price, stop_price, stop_dist, tp1_price, InpRR_TP1, tp2_price, InpRR_TP2
      );

      Print(msg);

      // 1. Play Sound Alarm on Laptop
      if(InpPlaySoundAlert)
         Alert(msg);

      // 2. Free Push Notification to MT5 Mobile Phone App
      if(InpSendPushNotification)
      {
         bool push_sent = SendNotification(msg);
         if(push_sent)
            Print("Push notification successfully sent to MT5 mobile app!");
         else
            Print("Push notification error: ", GetLastError(), ". Make sure MetaQuotes ID is filled in MT5 Options!");
      }

      // 3. Email (optional)
      if(InpSendEmailAlert)
         SendMail("BTC Buy Signal Alert", msg);
   }
   else
   {
      g_last_signal_time = current_bar_time;
   }
}
//+------------------------------------------------------------------+

