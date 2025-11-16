
#!/usr/bin/env python3
import os, csv, shutil, subprocess
from datetime import date, datetime, timedelta
import requests
from dotenv import load_dotenv



load_dotenv()
API_KEY = os.getenv("API_KEY")


# ── CONFIGURE ───────────────────────────────────────────────────────────────────
load_dotenv()
API_TOKEN  = os.getenv("MARKER")
GH_TOKEN   = os.getenv("GH_TOKEN")
GH_REPO    = os.getenv("GH_REPO")      # e.g. "youruser/flight-price-data"
CLONE_DIR  = "temp_flight_data"
CURRENCY   = "usd"
MARKET     = "us"
DAYS_AHEAD = 150

HOT_CITIES = [
    "NYC","LHR","DXB","TYO","SIN",
    "CDG","LAX","HND","SYD","MIA",
]


# ── STEP 1: COLLECT  ──────────────────────────────────────────────────────────────
def collect_flight_data(days_ahead=DAYS_AHEAD):
    today       = date.today()
    search_date = today.isoformat()
    cutoff      = today + timedelta(days=days_ahead)
    csvname     = f"flight_prices_{search_date}.csv"

    with open(csvname, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        # include search_date in header
        w.writerow([
            "origin",
            "destination",
            "search_date",
            "depart_date",
            "price"
        ])

        for origin in HOT_CITIES:
            for dest in HOT_CITIES:
                if origin == dest:
                    continue

                print(f"📡 {origin}→{dest}")
                url = (
                    "https://api.travelpayouts.com/aviasales/v3/get_latest_prices"
                    f"?origin={origin}&destination={dest}"
                    f"&currency={CURRENCY}&market={MARKET}"
                    "&period_type=year"
                    f"&token={API_TOKEN}"
                )

                try:
                    r = requests.get(url, timeout=10).json()
                    if r.get("success") and r.get("data"):
                        for item in r["data"]:
                            dep = datetime.fromisoformat(item["depart_date"]).date()
                            if dep <= cutoff:
                                # prepend search_date on each row
                                w.writerow([
                                    origin,
                                    dest,
                                    search_date,
                                    dep.isoformat(),
                                    item["value"]
                                ])
                    else:
                        print("  ⚠️ no data")
                except Exception as e:
                    print("  ❌", e)

    return csvname


# ── STEP 2: PUSH ─────────────────────────────────────────────────────────────────
def push_to_github(csvfile):
    repo_url = f"https://x-access-token:{GH_TOKEN}@github.com/{GH_REPO}.git"

    # clone or pull
    if os.path.isdir(os.path.join(CLONE_DIR, ".git")):
        subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=True)
    else:
        shutil.rmtree(CLONE_DIR, ignore_errors=True)
        subprocess.run(["git", "clone", repo_url, CLONE_DIR], check=True)

    # copy in the CSV
    dst = os.path.join(CLONE_DIR, csvfile)
    shutil.copy2(csvfile, dst)

    # commit only if there are changes
    status = subprocess.run(
        ["git", "-C", CLONE_DIR, "status", "--porcelain"],
        stdout=subprocess.PIPE, text=True, check=True
    ).stdout.strip()

    if not status:
        print("ℹ️  nothing to commit")
        return

    subprocess.run(["git", "-C", CLONE_DIR, "add", csvfile], check=True)
    subprocess.run(
        ["git", "-C", CLONE_DIR, "commit", "-m", f"📆 add {csvfile}"],
        check=True
    )
    subprocess.run(["git", "-C", CLONE_DIR, "push"], check=True)
    print("✅ pushed")


# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csvfile = collect_flight_data()
    push_to_github(csvfile)


