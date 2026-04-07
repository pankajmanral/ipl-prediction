import requests
import json
import os
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime

# --- CONFIGURATION ---
URL = "https://www.cricbuzz.com/cricket-series/9241/indian-premier-league-2026/matches"
JSON_FILE = "results_2026.json"
LIVE_JSON = "live_scores_2026.json"
PREDICT_SCRIPT = "predict_2026.py"

# --- RAPIDAPI CONFIG ---
RAPID_API_KEY = "b456313192msh64ecf2f7b1e9975p1aed90jsn265203a0cf4d"
RAPID_API_HOST = "cricket-live-line1.p.rapidapi.com"

NAME_MAP = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'RCB': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'PBKS': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'DC': 'Delhi Capitals',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'KKR': 'Kolkata Knight Riders',
    'Chennai Super Kings': 'Chennai Super Kings',
    'CSK': 'Chennai Super Kings',
    'Mumbai Indians': 'Mumbai Indians',
    'MI': 'Mumbai Indians',
    'Rajasthan Royals': 'Rajasthan Royals',
    'RR': 'Rajasthan Royals',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
    'SRH': 'Sunrisers Hyderabad',
    'Gujarat Titans': 'Gujarat Titans',
    'GT': 'Gujarat Titans',
    'Lucknow Super Giants': 'Lucknow Super Giants',
    'LSG': 'Lucknow Super Giants'
}

def clean_name(name):
    n = name.strip()
    return NAME_MAP.get(n, n)

def fetch_live_from_rapidapi():
    url = f"https://{RAPID_API_HOST}/liveMatches"
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": RAPID_API_HOST
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Handle the "moved to Advance" notice
            if not data.get("status") and "advance" in data.get("msg", "").lower():
                print(f"  [Notice] RapidAPI host {RAPID_API_HOST} is deprecated. Suggesting migration to: {data.get('msg')}")
                return {}

            if data.get("success") and data.get("data"):
                # Filter for IPL matches
                ipl_matches = [m for m in data["data"] if "IPL" in m.get("title", "").upper() or "INDIAN PREMIER LEAGUE" in m.get("title", "").upper()]
                if not ipl_matches and data["data"]:
                    m = data["data"][0]
                    t1 = m.get("team_a", "")
                    if any(team in t1 for team in NAME_MAP.keys()):
                        ipl_matches = [m]
                
                res = {}
                for m in ipl_matches:
                    match_id = m.get("match_id", "live")
                    res[match_id] = {
                        "status": m.get("match_status", "Live"),
                        "team1_score": m.get("team_a_score", "N/A"),
                        "team2_score": m.get("team_b_score", "Yet to bat"),
                        "team1": m.get("team_a"),
                        "team2": m.get("team_b"),
                        "title": m.get("title") or "IPL 2026 Live Match",
                        "batsmen": [
                            {"name": "Batter 1", "runs": 12, "balls": 8, "fours": 1, "sixes": 1},
                            {"name": "Batter 2", "runs": 4, "balls": 3, "fours": 0, "sixes": 0}
                        ],
                        "bowler": {"name": "Impact Bowler", "overs": 2.1, "runs": 15, "wickets": 1},
                        "crr": "8.4", "rrr": "9.2", "last_overs": ["1", "4", "0", "W", "2", "6"]
                    }
                return res
    except Exception as e:
        print(f"RapidAPI Error: {e}")
    return {}

def sync(simulate_live=False):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Syncing with RapidAPI & Cricbuzz...")
    
    # 1. Fetch live data from RapidAPI
    live_scores = fetch_live_from_rapidapi()
    print(f"  RapidAPI found {len(live_scores)} live matches.")

    # 2. Sync historical results from Cricbuzz 
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    soup = None
    try:
        response = requests.get(URL, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching Cricbuzz data: {e}")

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            local_results = json.load(f)
    else:
        local_results = []
    
    existing_mno = [m['match_no'] for m in local_results]
    new_data_found = False

    if soup:
        cards = soup.select('a.w-full.bg-cbWhite.flex.flex-col.p-3.gap-1')
        if not cards: cards = soup.select('a[href*="/cricket-scores/"]')
            
        for card in cards:
            header_el = card.select_one('div:nth-of-type(1) > span')
            if not header_el: continue
            h_text = header_el.text.strip()
            num_part = h_text.split('•')[0].strip().split()[0]
            mno_str = "".join(filter(str.isdigit, num_part))
            if not mno_str: continue
            mno = int(mno_str)
            
            if mno in existing_mno: continue
            
            result_div = card.find_next_sibling('div')
            result_text = result_div.text.strip() if result_div else ""
            if "won by" in result_text.lower() or "no result" in result_text.lower() or "abandoned" in result_text.lower():
                team_spans = card.select('div:nth-of-type(2) > div > div > span')
                score_spans = card.select('div:nth-of-type(2) > div > span')
                if len(team_spans) >= 2:
                    t1_name = clean_name(team_spans[0].text)
                    t1_score = score_spans[0].text.strip() if len(score_spans) > 0 else "N/A"
                    t2_name = clean_name(team_spans[1].text)
                    t2_score = score_spans[1].text.strip() if len(score_spans) > 1 else "N/A"
                    
                    winner = "None" if ("no result" in result_text.lower() or "abandoned" in result_text.lower()) else "Unknown"
                    if winner == "Unknown":
                        for team in NAME_MAP.values():
                            if team.lower() in result_text.lower():
                                winner = team; break
                    
                    local_results.append({
                        "match_no": mno, "home": t1_name, "away": t2_name, "winner": winner,
                        "home_score": t1_score if t1_score != t1_name else "N/A",
                        "away_score": t2_score if t2_score != t1_name else "N/A",
                        "margin": result_text
                    })
                    new_data_found = True
                    print(f"  + New Match Added: M#{mno} - {result_text}")

    if simulate_live:
        live_scores["sim_test"] = {
            "status": "KKR need 42 runs in 18 balls",
            "team1_score": "158/5 (17.2)",
            "team2_score": "199/6 (20)",
            "team1": "Kolkata Knight Riders",
            "team2": "Delhi Capitals",
            "title": "Match 28: KKR vs DC Live",
            "batsmen": [
                {"name": "Rinku Singh", "runs": 32, "balls": 14, "fours": 2, "sixes": 3},
                {"name": "Andre Russell", "runs": 15, "balls": 6, "fours": 1, "sixes": 1}
            ],
            "bowler": {"name": "Kuldeep Yadav", "overs": 3.2, "runs": 28, "wickets": 3},
            "crr": "9.11", "rrr": "14.0", "last_overs": ["6", "1", "W", "1", "4", "4"]
        }

    with open(LIVE_JSON, 'w') as f:
        json.dump(live_scores, f, indent=2)

    if new_data_found or (simulate_live or live_scores):
        if new_data_found:
            local_results = sorted(local_results, key=lambda x: x['match_no'])
            with open(JSON_FILE, 'w') as f: json.dump(local_results, f, indent=2)
        subprocess.run(["python3", PREDICT_SCRIPT])
        print("  Update Success.")
    else:
        print("  Status: No updates.")

if __name__ == "__main__":
    import sys
    sync(simulate_live="--simulate" in sys.argv)
