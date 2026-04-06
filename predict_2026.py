"""
IPL 2026 Enhanced Prediction Script - Updated with Live 2026 Data
Incorporates:
  - All 11 completed 2026 matches (actual results)
  - Current points table & 2026 team form
  - 2026 squad data (key players per team)
  - Individual player performance from 2026
  - Historical data (2008-2025) from IPL.csv
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── 2026 LIVE DATA ──────────────────────────────────────────────────────────
# All completed matches as of April 6, 2026
COMPLETED_2026 = [
    # match_no, home_team, away_team, winner, home_score, away_score, margin
    (1,  'Royal Challengers Bengaluru', 'Sunrisers Hyderabad',   'Royal Challengers Bengaluru', '203/4', '201/9',   '6 wkts'),
    (2,  'Mumbai Indians',              'Kolkata Knight Riders', 'Mumbai Indians',              '224/4', '220/4',   '6 wkts'),
    (3,  'Rajasthan Royals',            'Chennai Super Kings',   'Rajasthan Royals',            '128/2', '127',     '8 wkts'),
    (4,  'Punjab Kings',                'Gujarat Titans',        'Punjab Kings',                '165/7', '162/6',   '3 wkts'),
    (5,  'Lucknow Super Giants',        'Delhi Capitals',        'Delhi Capitals',              '141',   '145/4',   '6 wkts'),
    (6,  'Kolkata Knight Riders',       'Sunrisers Hyderabad',   'Sunrisers Hyderabad',         '161',   '226/8',   '65 runs'),
    (7,  'Chennai Super Kings',         'Punjab Kings',          'Punjab Kings',                '208/5', '209/5',   '5 wkts'),
    (8,  'Delhi Capitals',              'Mumbai Indians',        'Delhi Capitals',              '162/6', '165/4',   '6 wkts'),  # DC batted 1st
    (9,  'Rajasthan Royals',            'Gujarat Titans',        'Rajasthan Royals',            '210/6', '204/8',   '6 runs'),
    (10, 'Sunrisers Hyderabad',         'Lucknow Super Giants',  'Lucknow Super Giants',        '156/9', '160/5',   '5 wkts'),
    (11, 'Royal Challengers Bengaluru', 'Chennai Super Kings',   'Royal Challengers Bengaluru', '250/3', '207',     '43 runs'),
]

# Points table as of April 6, 2026 (after 11 matches)
POINTS_TABLE_2026 = {
    # team: [played, won, lost, points, nrr]
    'Royal Challengers Bengaluru': [2, 2, 0, 4,  +2.501],
    'Rajasthan Royals':            [2, 2, 0, 4,  +2.233],
    'Delhi Capitals':              [2, 2, 0, 4,  +1.170],
    'Punjab Kings':                [2, 2, 0, 4,  +0.637],
    'Sunrisers Hyderabad':         [3, 1, 2, 2,  +0.275],
    'Mumbai Indians':              [2, 1, 1, 2,  -0.206],
    'Lucknow Super Giants':        [2, 1, 1, 2,  -0.542],
    'Gujarat Titans':              [2, 0, 2, 0,  -0.424],
    'Kolkata Knight Riders':       [2, 0, 2, 0,  -1.964],
    'Chennai Super Kings':         [3, 0, 3, 0,  -2.517],
}

# 2026 Current Form: win rate in 2026
FORM_2026 = {
    'Royal Challengers Bengaluru': 2/2,   # 2W 0L
    'Rajasthan Royals':            2/2,   # 2W 0L
    'Delhi Capitals':              2/2,   # 2W 0L
    'Punjab Kings':                2/2,   # 2W 0L
    'Sunrisers Hyderabad':         1/3,   # 1W 2L
    'Mumbai Indians':              1/2,   # 1W 1L
    'Lucknow Super Giants':        1/2,   # 1W 1L
    'Gujarat Titans':              0/2,   # 0W 2L
    'Kolkata Knight Riders':       0/2,   # 0W 2L
    'Chennai Super Kings':         0/3,   # 0W 3L
}

# 2026 Key Players & Star Rating (based on IPL 2026 performance so far)
# Batting: runs, avg_score_per_match, strike_rate  
# Bowling: wickets, economy
SQUAD_2026 = {
    'Royal Challengers Bengaluru': {
        'captain': 'Rajat Patidar',
        'key_batters': ['Virat Kohli', 'Devdutt Padikkal', 'Tim David', 'Rajat Patidar', 'Phil Salt'],
        'key_bowlers': ['Bhuvneshwar Kumar', 'Josh Hazlewood', 'Jacob Duffy', 'Krunal Pandya'],
        'star_players': ['Virat Kohli', 'Tim David', 'Jacob Duffy', 'Bhuvneshwar Kumar'],
        'injuries': ['Josh Hazlewood (unavailable early)'],
        'batting_depth': 8.5,
        'bowling_strength': 8.0,
        'squad_strength': 8.5,
    },
    'Sunrisers Hyderabad': {
        'captain': 'Ishan Kishan',
        'key_batters': ['Travis Head', 'Heinrich Klaasen', 'Abhishek Sharma', 'Ishan Kishan'],
        'key_bowlers': ['David Payne', 'Mohammed Shami', 'Nitish Kumar Reddy', 'Jaydev Unadkat'],
        'star_players': ['Travis Head', 'Heinrich Klaasen', 'Ishan Kishan'],
        'injuries': ['Pat Cummins (back)', 'Jack Edwards (foot)'],
        'batting_depth': 8.0,
        'bowling_strength': 7.1,
        'squad_strength': 7.5,
    },
    'Mumbai Indians': {
        'captain': 'Hardik Pandya',
        'key_batters': ['Rohit Sharma', 'Suryakumar Yadav', 'Hardik Pandya', 'Tilak Varma'],
        'key_bowlers': ['Jasprit Bumrah', 'Trent Boult', 'Hardik Pandya', 'Deepak Chahar'],
        'star_players': ['Rohit Sharma', 'Suryakumar Yadav', 'Jasprit Bumrah', 'Trent Boult'],
        'injuries': ['Atharva Ankolekar (knee)'],
        'batting_depth': 8.5,
        'bowling_strength': 9.0,
        'squad_strength': 8.8,
    },
    'Kolkata Knight Riders': {
        'captain': 'Ajinkya Rahane',
        'key_batters': ['Sunil Narine', 'Rinku Singh', 'Cameron Green', 'Angkrish Raghuvanshi'],
        'key_bowlers': ['Varun Chakaravarthy', 'Saurabh Dubey', 'Andre Russell', 'Mitchell Starc'],
        'star_players': ['Sunil Narine', 'Varun Chakaravarthy', 'Cameron Green'],
        'injuries': ['Harshit Rana (knee)', 'Akash Deep (back)', 'Matheesha Pathirana (shoulder)', 'Mustafizur Rahman (unavailable)'],
        'batting_depth': 7.5,
        'bowling_strength': 7.3,
        'squad_strength': 7.4,
    },
    'Rajasthan Royals': {
        'captain': 'Sanju Samson',
        'key_batters': ['Yashasvi Jaiswal', 'Sanju Samson', 'Dhruv Jurel', 'Shimron Hetmyer'],
        'key_bowlers': ['Ravi Bishnoi', 'Jofra Archer', 'Yuzvendra Chahal', 'Nandre Burger'],
        'star_players': ['Yashasvi Jaiswal', 'Ravi Bishnoi', 'Jofra Archer', 'Ravindra Jadeja'],
        'injuries': ['Sam Curran (groin)'],
        'batting_depth': 8.0,
        'bowling_strength': 8.5,
        'squad_strength': 8.2,
    },
    'Chennai Super Kings': {
        'captain': 'Ruturaj Gaikwad',
        'key_batters': ['Ruturaj Gaikwad', 'Ayush Mhatre', 'MS Dhoni', 'Sarfaraz Khan'],
        'key_bowlers': ['Ravindra Jadeja', 'Anshul Kamboj', 'Spencer Johnson', 'Noor Ahmad'],
        'star_players': ['Ruturaj Gaikwad', 'Ravindra Jadeja'],
        'injuries': ['MS Dhoni (calf)', 'Nathan Ellis (hamstring)'],
        'batting_depth': 7.2,
        'bowling_strength': 7.0,
        'squad_strength': 7.1,
    },
    'Punjab Kings': {
        'captain': 'Shreyas Iyer',
        'key_batters': ['Shreyas Iyer', 'Priyansh Arya', 'Cooper Connolly', 'Shashank Singh'],
        'key_bowlers': ['Arshdeep Singh', 'Vijaykumar Vyshak', 'Marcus Stoinis', 'Yuvraj Singh'],
        'star_players': ['Shreyas Iyer', 'Arshdeep Singh', 'Cooper Connolly', 'Vijaykumar Vyshak'],
        'injuries': ['Lockie Ferguson (personal)'],
        'batting_depth': 7.8,
        'bowling_strength': 7.9,
        'squad_strength': 7.9,
    },
    'Gujarat Titans': {
        'captain': 'Shubman Gill',
        'key_batters': ['Shubman Gill', 'Jos Buttler', 'Sai Sudharsan', 'Washington Sundar'],
        'key_bowlers': ['Rashid Khan', 'Kagiso Rabada', 'Mohammed Siraj', 'Prasidh Krishna'],
        'star_players': ['Shubman Gill', 'Rashid Khan', 'Jos Buttler', 'Kagiso Rabada'],
        'injuries': ['Prithviraj Yarra (unavailable)'],
        'batting_depth': 8.0,
        'bowling_strength': 8.5,
        'squad_strength': 8.0,
    },
    'Lucknow Super Giants': {
        'captain': 'Rishabh Pant',
        'key_batters': ['Rishabh Pant', 'Mitchell Marsh', 'Abdul Samad', 'Nicholas Pooran'],
        'key_bowlers': ['Mohammed Shami', 'T Natarajan', 'Ravi Bishnoi', 'Avesh Khan'],
        'star_players': ['Rishabh Pant', 'Mohammed Shami', 'Mitchell Marsh'],
        'injuries': ['Wanindu Hasaranga (hamstring)', 'Josh Inglis (personal)'],
        'batting_depth': 7.6,
        'bowling_strength': 7.7,
        'squad_strength': 7.6,
    },
    'Delhi Capitals': {
        'captain': 'Axar Patel',
        'key_batters': ['KL Rahul', 'Sameer Rizvi', 'Tristan Stubbs', 'Faf du Plessis'],
        'key_bowlers': ['Khaleel Ahmed', 'Kuldeep Yadav', 'Axar Patel', 'Lungi Ngidi'],
        'star_players': ['KL Rahul', 'Sameer Rizvi', 'Kuldeep Yadav'],
        'injuries': ['Mitchell Starc (workload mgmt)'],
        'batting_depth': 8.0,
        'bowling_strength': 8.1,
        'squad_strength': 8.0,
    },
}

# 2026 Player performance (runs/wickets so far)
TOP_BATTERS_2026 = {
    'Sameer Rizvi':      {'team': 'Delhi Capitals',              'runs': 160, 'matches': 2, 'sr': 176},
    'Heinrich Klaasen':  {'team': 'Sunrisers Hyderabad',         'runs': 145, 'matches': 3, 'sr': 168},
    'Rohit Sharma':      {'team': 'Mumbai Indians',              'runs': 113, 'matches': 2, 'sr': 158},
    'Devdutt Padikkal':  {'team': 'Royal Challengers Bengaluru', 'runs': 111, 'matches': 2, 'sr': 165},
    'Cooper Connolly':   {'team': 'Punjab Kings',                'runs': 108, 'matches': 2, 'sr': 163},
    'Tim David':         {'team': 'Royal Challengers Bengaluru', 'runs': 95,  'matches': 2, 'sr': 240},
    'Yashasvi Jaiswal':  {'team': 'Rajasthan Royals',            'runs': 90,  'matches': 2, 'sr': 185},
    'Rishabh Pant':      {'team': 'Lucknow Super Giants',        'runs': 85,  'matches': 2, 'sr': 142},
    'Shreyas Iyer':      {'team': 'Punjab Kings',                'runs': 80,  'matches': 2, 'sr': 152},
}

TOP_BOWLERS_2026 = {
    'Ravi Bishnoi':        {'team': 'Rajasthan Royals',            'wickets': 5, 'matches': 2, 'eco': 7.2},
    'Vijaykumar Vyshak':   {'team': 'Punjab Kings',                'wickets': 5, 'matches': 2, 'eco': 8.1},
    'Jacob Duffy':         {'team': 'Royal Challengers Bengaluru', 'wickets': 5, 'matches': 2, 'eco': 7.8},
    'Anshul Kamboj':       {'team': 'Chennai Super Kings',         'wickets': 5, 'matches': 3, 'eco': 9.1},
    'Bhuvneshwar Kumar':   {'team': 'Royal Challengers Bengaluru', 'wickets': 4, 'matches': 2, 'eco': 7.5},
    'Mohammed Shami':      {'team': 'Lucknow Super Giants',        'wickets': 4, 'matches': 2, 'eco': 6.8},
    'Lungi Ngidi':         {'team': 'Delhi Capitals',              'wickets': 4, 'matches': 2, 'eco': 8.2},
    'T Natarajan':         {'team': 'Delhi Capitals',              'wickets': 3, 'matches': 2, 'eco': 7.9},
    'Nitish Kumar Reddy':  {'team': 'Sunrisers Hyderabad',         'wickets': 3, 'matches': 3, 'eco': 8.5},
}

TEAMS_2026 = list(SQUAD_2026.keys())

# ─── HISTORICAL DATA ─────────────────────────────────────────────────────────
print("Loading IPL historical data (2008–2025)...")
df = pd.read_csv('IPL.csv', low_memory=False)

rename = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
}

match_cols = ['match_id','date','batting_team','bowling_team',
              'match_won_by','toss_winner','toss_decision','venue','city','season']
mdf = df[match_cols].drop_duplicates('match_id').copy()
mdf['date'] = pd.to_datetime(mdf['date'])
mdf = mdf[mdf['match_won_by'] != 'Unknown'].dropna(subset=['match_won_by'])

for col in ['batting_team','bowling_team','match_won_by','toss_winner']:
    mdf[col] = mdf[col].replace(rename)

match_teams = mdf.groupby('match_id').first().reset_index()
match_teams = match_teams.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2'})
match_teams['team1_won'] = (match_teams['match_won_by'] == match_teams['team1']).astype(int)

valid = match_teams[
    match_teams['team1'].isin(TEAMS_2026) &
    match_teams['team2'].isin(TEAMS_2026)
].copy()

print(f"  Historical matches between 2026 teams: {len(valid)}")

# Add 2026 actual results to the dataset for boosts
extra_rows = []
for m in COMPLETED_2026:
    match_no, home, away, winner, score_home, score_away, margin = m
    t1_won = 1 if winner == home else 0
    extra_rows.append({
        'match_id': f'2026_{match_no}',
        'date': pd.Timestamp('2026-03-28') + pd.Timedelta(days=match_no-1),
        'team1': home, 'team2': away,
        'match_won_by': winner, 'venue': '',
        'toss_winner': '', 'toss_decision': '',
        'city': '', 'season': '2026',
        'team1_won': t1_won
    })

valid_with_2026 = pd.concat([valid, pd.DataFrame(extra_rows)], ignore_index=True)
valid_sorted = valid_with_2026.sort_values('date')

# ─── FEATURE COMPUTATION ─────────────────────────────────────────────────────
def compute_h2h(df_hist, team_a, team_b):
    matches = df_hist[
        ((df_hist['team1']==team_a) & (df_hist['team2']==team_b)) |
        ((df_hist['team1']==team_b) & (df_hist['team2']==team_a))
    ]
    if len(matches)==0: return 0.5, 0
    wins_a = len(matches[(matches['team1']==team_a)&(matches['team1_won']==1)]) + \
             len(matches[(matches['team2']==team_a)&(matches['team1_won']==0)])
    return wins_a/len(matches), len(matches)

def overall_win_rate(df_hist, team):
    t = df_hist[(df_hist['team1']==team)|(df_hist['team2']==team)]
    if len(t)==0: return 0.5
    w = len(t[(t['team1']==team)&(t['team1_won']==1)]) + \
        len(t[(t['team2']==team)&(t['team1_won']==0)])
    return w/len(t)

def recent_form(df_hist, team, n=20):
    t = df_hist[(df_hist['team1']==team)|(df_hist['team2']==team)].tail(n)
    if len(t)==0: return 0.5
    w = len(t[(t['team1']==team)&(t['team1_won']==1)]) + \
        len(t[(t['team2']==team)&(t['team1_won']==0)])
    return w/len(t)

def venue_win_rate(df_hist, team, venue):
    v = df_hist[(df_hist['venue']==venue)&
                ((df_hist['team1']==team)|(df_hist['team2']==team))]
    if len(v)==0: return overall_win_rate(df_hist, team)
    w = len(v[(v['team1']==team)&(v['team1_won']==1)]) + \
        len(v[(v['team2']==team)&(v['team1_won']==0)])
    return w/len(v)

def toss_bat_first_win_rate(df_hist, venue):
    v = df_hist[df_hist['venue']==venue]
    if len(v)==0: return 0.5
    return len(v[(v['toss_decision']=='bat')&(v['team1_won']==1)])/len(v)

HOME_VENUES = {
    'Royal Challengers Bengaluru': ['M Chinnaswamy Stadium','Shaheed Veer Narayan Singh International Cricket S'],
    'Sunrisers Hyderabad':         ['Rajiv Gandhi International Stadium'],
    'Mumbai Indians':              ['Wankhede Stadium','Dr DY Patil Sports Academy'],
    'Kolkata Knight Riders':       ['Eden Gardens'],
    'Rajasthan Royals':            ['Sawai Mansingh Stadium'],
    'Chennai Super Kings':         ['MA Chidambaram Stadium'],
    'Punjab Kings':                ['New International Cricket Stadium','Himachal Pradesh Cricket Association Stadium'],
    'Gujarat Titans':              ['Narendra Modi Stadium'],
    'Lucknow Super Giants':        ['Bharat Ratna Shri Atal Bihari Vajpayee Ekana Crick'],
    'Delhi Capitals':              ['Arun Jaitley Stadium'],
}

# Pre-compute base metrics on full dataset
wr_map      = {t: overall_win_rate(valid_sorted, t) for t in TEAMS_2026}
hist_form   = {t: recent_form(valid_sorted, t, 20) for t in TEAMS_2026}

# 2026 live form & squad strength adjustments
def get_effective_form(team):
    """Blend historical form (40%) + 2026 live form (60%)"""
    hist = hist_form.get(team, 0.5)
    live = FORM_2026.get(team, 0.5)
    return 0.4 * hist + 0.6 * live

def get_squad_strength(team):
    """Normalised squad strength from 2026 squad data"""
    s = SQUAD_2026.get(team, {})
    return s.get('squad_strength', 7.0) / 10.0

def get_2026_momentum(team):
    """Win rate in 2026 matches specifically"""
    pt = POINTS_TABLE_2026.get(team, [0,0,0,0,0])
    played = pt[0]
    if played == 0: return 0.5
    return pt[1] / played

def build_features(row):
    t1, t2, venue = row['team1'], row['team2'], str(row.get('venue',''))
    h2h_rate, h2h_n = compute_h2h(valid_sorted, t1, t2)

    feats = {
        # Historical win rates
        'team1_overall_wr':      wr_map.get(t1, 0.5),
        'team2_overall_wr':      wr_map.get(t2, 0.5),
        # Blended form (historical + 2026)
        'team1_eff_form':        get_effective_form(t1),
        'team2_eff_form':        get_effective_form(t2),
        # 2026 live momentum
        'team1_2026_momentum':   get_2026_momentum(t1),
        'team2_2026_momentum':   get_2026_momentum(t2),
        # Squad strength
        'team1_squad_str':       get_squad_strength(t1),
        'team2_squad_str':       get_squad_strength(t2),
        # H2H
        'h2h_team1_wr':          h2h_rate,
        'h2h_n_matches':         h2h_n,
        # Venue
        'team1_venue_wr':        venue_win_rate(valid_sorted, t1, venue),
        'team2_venue_wr':        venue_win_rate(valid_sorted, t2, venue),
        'team1_home':            int(venue in HOME_VENUES.get(t1, [])),
        'team2_home':            int(venue in HOME_VENUES.get(t2, [])),
        # NRR advantage (2026 season health)
        'team1_nrr':             POINTS_TABLE_2026.get(t1, [0,0,0,0,0])[4],
        'team2_nrr':             POINTS_TABLE_2026.get(t2, [0,0,0,0,0])[4],
        # Difference features
        'wr_diff':               wr_map.get(t1,0.5) - wr_map.get(t2,0.5),
        'form_diff':             get_effective_form(t1) - get_effective_form(t2),
        'momentum_diff':         get_2026_momentum(t1) - get_2026_momentum(t2),
        'squad_diff':            get_squad_strength(t1) - get_squad_strength(t2),
        'venue_wr_diff':         venue_win_rate(valid_sorted,t1,venue) - venue_win_rate(valid_sorted,t2,venue),
        'nrr_diff':              POINTS_TABLE_2026.get(t1,[0,0,0,0,0])[4] - POINTS_TABLE_2026.get(t2,[0,0,0,0,0])[4],
    }
    return feats

# ─── TRAIN MODELS ────────────────────────────────────────────────────────────
print("Building training features...")
X_rows, y_rows = [], []
for _, row in valid_sorted.iterrows():
    try:
        X_rows.append(build_features(row))
        y_rows.append(row['team1_won'])
    except: pass

X = pd.DataFrame(X_rows)
y = pd.Series(y_rows)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.04, max_depth=5, random_state=42)
lr = LogisticRegression(max_iter=1000, C=0.5)

rf.fit(X_train, y_train); rf_acc = accuracy_score(y_test, rf.predict(X_test))
gb.fit(X_train, y_train); gb_acc = accuracy_score(y_test, gb.predict(X_test))
lr.fit(X_train, y_train); lr_acc = accuracy_score(y_test, lr.predict(X_test))

print(f"  RF: {rf_acc:.3f}  GB: {gb_acc:.3f}  LR: {lr_acc:.3f}")

total_acc = rf_acc + gb_acc + lr_acc

def predict_match(home_team, away_team, venue):
    row = {'team1': home_team, 'team2': away_team, 'venue': venue}
    feats = pd.DataFrame([build_features(row)])
    p_rf = rf.predict_proba(feats)[0][1]
    p_gb = gb.predict_proba(feats)[0][1]
    p_lr = lr.predict_proba(feats)[0][1]
    p_team1 = (p_rf*rf_acc + p_gb*gb_acc + p_lr*lr_acc) / total_acc
    winner   = home_team if p_team1 >= 0.5 else away_team
    conf     = p_team1 if p_team1 >= 0.5 else 1-p_team1
    return winner, round(conf*100, 1), round(p_team1*100, 1)

# ─── LOAD SCHEDULE & PREDICT ─────────────────────────────────────────────────
print("\nGenerating 2026 predictions...")
schedule = pd.read_csv('ipl-2026-UTC.csv')
schedule.columns = schedule.columns.str.strip()
schedule['Date'] = pd.to_datetime(schedule['Date'], format='%d/%m/%Y %H:%M')

completed_nos = {m[0] for m in COMPLETED_2026}
completed_lookup = {m[0]: m for m in COMPLETED_2026}

results = []
for _, row in schedule.iterrows():
    home  = str(row['Home Team']).strip()
    away  = str(row['Away Team']).strip()
    venue = str(row['Location']).strip()
    date  = row['Date']
    mno   = int(row['Match Number'])
    rno   = int(row['Round Number'])

    if home not in TEAMS_2026 or away not in TEAMS_2026: continue

    is_completed = mno in completed_nos
    actual_winner = completed_lookup[mno][3] if is_completed else None

    winner, confidence, home_prob = predict_match(home, away, venue)
    away_prob = round(100 - home_prob, 1)
    h2h_rate, h2h_n = compute_h2h(valid_sorted, home, away)

    pt_home = POINTS_TABLE_2026.get(home, [0,0,0,0,0])
    pt_away = POINTS_TABLE_2026.get(away, [0,0,0,0,0])

    results.append({
        'match_no':         mno,
        'round_no':         rno,
        'date':             date.strftime('%d %b %Y'),
        'day':              date.strftime('%A'),
        'time_utc':         date.strftime('%I:%M %p') + ' UTC',
        'venue':            venue,
        'home_team':        home,
        'away_team':        away,
        'predicted_winner': winner,
        'confidence':       confidence,
        'home_win_prob':    home_prob,
        'away_win_prob':    away_prob,
        'is_completed':     is_completed,
        'actual_winner':    actual_winner,
        'prediction_correct': (winner == actual_winner) if is_completed else None,
        # Form & stats
        'home_2026_form':   round(get_effective_form(home)*100, 1),
        'away_2026_form':   round(get_effective_form(away)*100, 1),
        'home_2026_wins':   pt_home[1],
        'away_2026_wins':   pt_away[1],
        'home_2026_played': pt_home[0],
        'away_2026_played': pt_away[0],
        'home_nrr':         pt_home[4],
        'away_nrr':         pt_away[4],
        'home_squad_str':   round(get_squad_strength(home)*10, 1),
        'away_squad_str':   round(get_squad_strength(away)*10, 1),
        'home_momentum':    round(get_2026_momentum(home)*100, 1),
        'away_momentum':    round(get_2026_momentum(away)*100, 1),
        'h2h_matches':      h2h_n,
        'h2h_home_wr':      round(h2h_rate*100, 1),
        'home_overall_wr':  round(wr_map.get(home, 0.5)*100, 1),
        'away_overall_wr':  round(wr_map.get(away, 0.5)*100, 1),
        'home_captain':     SQUAD_2026.get(home,{}).get('captain',''),
        'away_captain':     SQUAD_2026.get(away,{}).get('captain',''),
        'home_stars':       SQUAD_2026.get(home,{}).get('star_players',[]),
        'away_stars':       SQUAD_2026.get(away,{}).get('star_players',[]),
    })

# Check accuracy on completed matches
completed_results = [r for r in results if r['is_completed']]
correct = sum(1 for r in completed_results if r['prediction_correct'])
acc_on_2026 = round(correct / len(completed_results) * 100, 1) if completed_results else 0

# Player leaderboards
top_batters_list = sorted(
    [{'name': k, **v} for k, v in TOP_BATTERS_2026.items()],
    key=lambda x: x['runs'], reverse=True
)
top_bowlers_list = sorted(
    [{'name': k, **v} for k, v in TOP_BOWLERS_2026.items()],
    key=lambda x: x['wickets'], reverse=True
)

output = {
    'generated_at':    '2026-04-06',
    'model_accuracy':  round((rf_acc + gb_acc + lr_acc)/3*100, 1),
    'rf_accuracy':     round(rf_acc*100, 1),
    'gb_accuracy':     round(gb_acc*100, 1),
    'lr_accuracy':     round(lr_acc*100, 1),
    'acc_on_2026':     acc_on_2026,
    'correct_on_2026': correct,
    'total_completed': len(completed_results),
    'historical_matches': len(valid_sorted),
    'completed_matches': [
        {'match_no': m[0], 'home': m[1], 'away': m[2], 'winner': m[3],
         'home_score': m[4], 'away_score': m[5], 'margin': m[6]}
        for m in COMPLETED_2026
    ],
    'points_table': [
        {'team': t, 'played': v[0], 'won': v[1], 'lost': v[2],
         'points': v[3], 'nrr': v[4]}
        for t, v in sorted(POINTS_TABLE_2026.items(), key=lambda x: (-x[1][3], -x[1][4]))
    ],
    'top_batters': top_batters_list,
    'top_bowlers': top_bowlers_list,
    'squads': {
        t: {
            'captain': d['captain'],
            'key_batters': d['key_batters'],
            'key_bowlers': d['key_bowlers'],
            'star_players': d['star_players'],
            'injuries': d['injuries'],
            'batting_depth': d['batting_depth'],
            'bowling_strength': d['bowling_strength'],
            'squad_strength': d['squad_strength'],
        }
        for t, d in SQUAD_2026.items()
    },
    'matches': results
}

with open('predictions_2026.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved predictions_2026.json")
print(f"   Total predictions: {len(results)}")
print(f"   Completed matches: {len(completed_results)}")
print(f"   Prediction accuracy on 2026 matches: {correct}/{len(completed_results)} = {acc_on_2026}%")
print()

# Print completed match review
print("=" * 80)
print("2026 COMPLETED MATCHES — ACTUAL vs PREDICTED")
print("=" * 80)
for r in completed_results:
    status = '✅' if r['prediction_correct'] else '❌'
    print(f" {status} M{r['match_no']:>2} | {r['date']} | {r['home_team']:>30} vs {r['away_team']:<30} | Actual: {r['actual_winner']:<30} | Pred: {r['predicted_winner']}")

print()
print("REMAINING MATCH PREDICTIONS (upcoming):")
print("=" * 80)
for r in [x for x in results if not x['is_completed']]:
    hw = '🏠' if r['predicted_winner']==r['home_team'] else '✈️'
    print(f" M{r['match_no']:>2} | {r['date']:>12} | {r['home_team']:>30} vs {r['away_team']:<30} | {hw} {r['predicted_winner']:<30} | {r['confidence']}%")
