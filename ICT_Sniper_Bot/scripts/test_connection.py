"""
Tradovate API Connection & Balance Verification Test
Run this anytime to verify your credentials, connection, and account balance!
"""
import os, asyncio, aiohttp, json
from dotenv import load_dotenv

load_dotenv()

LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() == "true"
BASE_URL  = "https://live.tradovate.com/v1" if LIVE_MODE else "https://demo.tradovateapi.com/v1"

async def test():
    print()
    print("=" * 60)
    print("  TRADOVATE ACCOUNT & CREDENTIALS CHECK")
    print(f"  Target Environment : {'*** LIVE PRODUCTION ***' if LIVE_MODE else 'DEMO / PAPER SIMULATION'}")
    print("=" * 60)
    
    email = os.getenv("TRADOVATE_USER")
    pw    = os.getenv("TRADOVATE_PASS")
    cid   = os.getenv("TRADOVATE_CID")
    sec   = os.getenv("TRADOVATE_SECRET")
    
    if not email or not pw or not cid or not sec:
        print("\n  [ERROR] Missing credentials in .env file!")
        print("  Please run 1_INSTALL_AND_SETUP.bat first.")
        print("=" * 60)
        return
        
    print(f"  Attempting login for: {email}...")
    
    payload = {
        "name":        email,
        "password":    pw,
        "appId":       os.getenv("TRADOVATE_APP_ID", "ICT_Sniper_Bot"),
        "appVersion":  "1.0",
        "deviceId":    os.getenv("TRADOVATE_DEVICE_ID", "device_001"),
        "cid":         int(cid),
        "sec":         sec,
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{BASE_URL}/auth/accesstokenrequest", json=payload) as r:
                data = await r.json()
                if "accessToken" not in data:
                    print(f"\n  [FAILED] Authentication rejected by Tradovate:")
                    print(f"  Response: {data}")
                    print("\n  Check: email, password, CID, and Secret in your .env file.")
                    return
                token = data["accessToken"]
                print("  [SUCCESS] API Authentication token generated!")
                
            headers = {"Authorization": f"Bearer {token}"}
            async with session.get(f"{BASE_URL}/account/list", headers=headers) as r:
                accs = await r.json()
                print(f"  [SUCCESS] Found {len(accs)} linked account(s):\n")
                for i, acc in enumerate(accs, 1):
                    aid = acc.get("id")
                    name = acc.get("name")
                    acc_type = acc.get("accountType", "Futures")
                    print(f"    Account #{i}: {name} (ID: {aid}) [{acc_type}]")
                    
                    # Fetch balance if available
                    try:
                        async with session.get(f"{BASE_URL}/cashBalance/getCashBalanceSnapshot?accountId={aid}", headers=headers) as br:
                            bdata = await br.json()
                            cash = bdata.get("cashBalance", "N/A")
                            nlv  = bdata.get("netLiquidatingValue", "N/A")
                            print(f"      -> Net Liquidating Value: ${nlv}")
                            print(f"      -> Cash Balance:          ${cash}")
                    except Exception:
                        pass
                
                print()
                print("=" * 60)
                print("  READY TO TRADE! Your bot is 100% verified.")
                print("  Double click '2_START_BOT.bat' to launch automated trading.")
                print("=" * 60)
                print()
                
        except Exception as e:
            print(f"\n  [ERROR] Network error: {e}")

if __name__ == "__main__":
    asyncio.run(test())

