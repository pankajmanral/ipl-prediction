import requests
import json
import os
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime

# --- CONFIGURATION ---
URL = "https://www.cricbuzz.com/cricket-series/9241/indian-premier-league-2026/matches"
JSON_FILE = "results_2026.json"
PREDICT_SCRIPT = "predict_2026.py"

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

def sync():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to Cricbuzz Live Schedule...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(URL, headers=headers, timeout=15)
        print(f"HTTP Status: {response.status_code}")
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Load existing results
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            local_results = json.load(f)
    else:
        local_results = []
    
    existing_mno = [m['match_no'] for m in local_results]
    
    # Corrected selectors for Cricbuzz current structure
    # Based on browser analysis
    cards = soup.select('a.w-full.bg-cbWhite.flex.flex-col.p-3.gap-1')
    if not cards:
        # Fallback to a broader selector if classes changed or were misinterpreted
        cards = soup.select('a[href*="/cricket-scores/"]')
        
    print(f"Found {len(cards)} match cards on the page.")
    new_data_found = False

    for card in cards:
        # Header (e.g. "12th Match")
        header_el = card.select_one('div:nth-of-type(1) > span')
        if not header_el: continue
        h_text = header_el.text.strip()
        
        # Parse match number
        mno_str = "".join(filter(str.isdigit, h_text.split('•')[0].strip().split()[0]))
        if not mno_str: continue
        mno = int(mno_str)
        
        if mno in existing_mno: continue
        
        # Result text is in a sibling div (Cricbuzz design)
        result_div = card.find_next_sibling('div')
        result_text = result_div.text.strip() if result_div else ""
        
        is_no_result = "no result" in result_text.lower() or "abandoned" in result_text.lower()
        
        if "won by" in result_text.lower() or is_no_result:
            # Teams and Scores
            team_spans = card.select('div:nth-of-type(2) > div > div > span')
            score_spans = card.select('div:nth-of-type(2) > div > span')
            
            if len(team_spans) >= 2:
                t1_name = clean_name(team_spans[0].text)
                t1_score = score_spans[0].text.strip() if len(score_spans) > 0 else "N/A"
                
                t2_name = clean_name(team_spans[1].text)
                t2_score = score_spans[1].text.strip() if len(score_spans) > 1 else "N/A"
                
                winner = "None" if is_no_result else "Unknown"
                if not is_no_result:
                    for team in NAME_MAP.values():
                        if team.lower() in result_text.lower():
                            winner = team
                            break
                
                new_entry = {
                    "match_no": mno,
                    "home": t1_name,
                    "away": t2_name,
                    "winner": winner,
                    "home_score": t1_score if t1_score != t1_name else "N/A",
                    "away_score": t2_score if t2_score != t2_name else "N/A",
                    "margin": result_text if is_no_result else result_text.split("won by")[-1].strip()
                }
                
                local_results.append(new_entry)
                new_data_found = True
                print(f"  + New Match Added: M#{mno} - {result_text}")

    if new_data_found:
        local_results = sorted(local_results, key=lambda x: x['match_no'])
        with open(JSON_FILE, 'w') as f:
            json.dump(local_results, f, indent=2)
        
        print("  Regenerating AI Prediction Hub Dashboard...")
        subprocess.run(["python3", PREDICT_SCRIPT])
        print("  Update Success: Dashboard is now synchronous with live results.")
    else:
        print("  Status: No new completed matches available to sync yet.")

if __name__ == "__main__":
    sync()
