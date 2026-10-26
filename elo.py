import json
import math
import re
import os
import time
from curl_cffi import requests

# ============================================================
# CONFIG
# ============================================================
CACHE_FILE = "elo_cache.json"

# Elo Settings
K_FACTOR = 20
HOME_ADVANTAGE = 65
INITIAL_ELO = 1500

LEAGUES = {
    "EPL": "en.1", "Championship": "en.2", 
    "La Liga": "es.1", "Bundesliga": "de.1", 
    "Serie A": "it.1", "Ligue 1": "fr.1"
}
SEASONS = ["2024-25", "2023-24", "2022-23"]

# ============================================================
# TEAM NORMALIZER
# ============================================================
NOISE_WORDS = {"fc","afc","cf","sc","ac","as","ss","ssc","sv","vfl","vfb","bsc","tsg",
    "fsv","bk","sk","fk","kf","nk","hnk","cd","sd","ud","rc","rcd","cs","us","club",
    "de","futbol","football","sport","sportif","calcio","soccer","the"}

TEAM_ALIASES = {
    "man city":"manchester city","man utd":"manchester united","man united":"manchester united",
    "newcastle":"newcastle united","west ham":"west ham united","brighton":"brighton & hove albion",
    "tottenham":"tottenham hotspur","spurs":"tottenham hotspur","wolves":"wolverhampton wanderers",
    "nottm forest":"nottingham forest","nott'm forest":"nottingham forest","nottingham":"nottingham forest",
    "sheffield utd":"sheffield united","sheffield weds":"sheffield wednesday","leicester":"leicester city",
    "leeds":"leeds united","norwich":"norwich city","stoke":"stoke city","swansea":"swansea city",
    "hull":"hull city","cardiff":"cardiff city","birmingham":"birmingham city","barca":"barcelona",
    "atletico":"atletico madrid","real":"real madrid","betis":"real betis","sociedad":"real sociedad",
    "athletic club":"athletic bilbao","celta":"celta vigo","rayo":"rayo vallecano","bayern":"bayern munich",
    "dortmund":"borussia dortmund","bvb":"borussia dortmund","leverkusen":"bayer leverkusen",
    "leipzig":"rb leipzig","frankfurt":"eintracht frankfurt","wolfsburg":"vfl wolfsburg",
    "gladbach":"borussia monchengladbach","freiburg":"sc freiburg","hoffenheim":"tsg hoffenheim",
    "mainz":"mainz 05","augsburg":"fc augsburg","stuttgart":"vfb stuttgart","werder":"werder bremen",
    "union berlin":"union berlin","bochum":"vfl bochum","heidenheim":"1. fc heidenheim",
    "darmstadt":"sv darmstadt 98","koln":"1. fc koln","inter":"inter milan","milan":"ac milan",
    "juve":"juventus","roma":"as roma","lazio":"ss lazio","verona":"hellas verona",
    "psg":"paris saint-germain","marseille":"olympique marseille","lyon":"olympique lyonnais",
    "monaco":"as monaco","lens":"rc lens","nice":"ogc nice","rennes":"stade rennais",
    "reims":"stade de reims","strasbourg":"rc strasbourg","brest":"stade brestois",
    "metz":"fc metz","lorient":"fc lorient","clermont":"clermont foot","az":"az alkmaar",
    "twente":"fc twente","utrecht":"fc utrecht","sporting":"sporting cp","braga":"sc braga",
}

def normalize(name):
    if not name: return ""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s]"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    words = [t for t in s.split() if t not in NOISE_WORDS]
    clean_name = " ".join(words)
    return TEAM_ALIASES.get(clean_name, clean_name)

# ============================================================
# ELO MATH
# ============================================================
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(ratings, home, away, home_goals, away_goals):
    r_home = ratings.get(home, INITIAL_ELO)
    r_away = ratings.get(away, INITIAL_ELO)

    r_home_adj = r_home + HOME_ADVANTAGE
    we_home = expected_score(r_home_adj, r_away)
    
    if home_goals > away_goals:
        w_home = 1.0
    elif home_goals < away_goals:
        w_home = 0.0
    else:
        w_home = 0.5

    goal_diff = abs(home_goals - away_goals)
    mov = 1.0
    if goal_diff == 2: mov = 1.5
    elif goal_diff >= 3: mov = (11 + goal_diff) / 8.0

    change = K_FACTOR * mov * (w_home - we_home)
    ratings[home] = r_home + change
    ratings[away] = r_away - change

def predict_1x2(rating_home, rating_away):
    r_home_adj = rating_home + HOME_ADVANTAGE
    dr = r_home_adj - rating_away
    
    we_home = 1 / (1 + 10 ** (-dr / 400))
    p_draw = 0.30 * math.exp(-abs(dr) / 200)
    
    p_home = we_home - (p_draw / 2)
    p_away = (1.0 - we_home) - (p_draw / 2)

    p_home = max(0.01, p_home)
    p_away = max(0.01, p_away)
    
    total = p_home + p_draw + p_away
    return p_home/total, p_draw/total, p_away/total

# ============================================================
# DATA FETCHING
# ============================================================
def train_elo():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    print("Training Elo Model on 3 years of historical data (1-2 mins)...")
    all_matches = []
    
    for season in SEASONS:
        for l_name, l_code in LEAGUES.items():
            url = f"https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{l_code}.json"
            try:
                r = requests.get(url, timeout=10, verify=False)
                if r.status_code == 200:
                    for m in r.json().get("matches", []):
                        if m.get("score", {}).get("ft"):
                            all_matches.append({
                                "date": m["date"],
                                "home": normalize(m.get("team1")),
                                "away": normalize(m.get("team2")),
                                "hg": m["score"]["ft"][0],
                                "ag": m["score"]["ft"][1]
                            })
            except: pass
            
    all_matches.sort(key=lambda x: x["date"])
    
    ratings = {}
    for m in all_matches:
        update_elo(ratings, m["home"], m["away"], m["hg"], m["ag"])
        
    with open(CACHE_FILE, "w") as f:
        json.dump(ratings, f)
        
    return ratings

def get_upcoming_betika_matches():
    print("Fetching upcoming games from Betika...")
    session = requests.Session()
    session.verify = False
    
    try: session.get("https://www.betika.com/en-ke/", impersonate="chrome120", timeout=10)
    except: pass
    
    url = "https://api.betika.com/v1/uo/matches?tab=upcoming&limit=1000"
    
    try:
        r = session.get(url, impersonate="chrome120", timeout=15)
        data = r.json()
        
        all_matches = data.get("data", [])
        print(f"   [DEBUG] Betika returned {len(all_matches)} total matches.")
        
        target_matches = []
        for m in all_matches:
            comp_name = str(m.get("competition_name", "")).lower()
            cat = str(m.get("category", "")).lower()
            
            if ("premier league" in comp_name and "england" in cat) or \
               ("championship" in comp_name and "england" in cat) or \
               ("champions league" in comp_name):
                target_matches.append(m)
                
        return target_matches
    except Exception as e:
        print(f"Error fetching Betika: {e}")
        return []

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" 🏆 PREMIUM ELO MODEL: BETIKA (EPL / CHAMPIONSHIP / UCL) 🏆")
    print("=" * 80)

    ratings = train_elo()
    matches = get_upcoming_betika_matches()

    if not matches:
        print("\n❌ No upcoming games found on Betika.")
        print("💡 TIP: Turn on a VPN or use your phone's hotspot to change your IP and run it again.")
        exit()

    print(f"\nFound {len(matches)} target matches. Generating Elo Predictions...\n")
    print("=" * 80)
    print(f"{'Match':<30} | {'1 (Home)':<10} | {'X (Draw)':<10} | {'2 (Away)':<10} | {'Top Pick'}")
    print("-" * 80)

    for m in matches:
        home_raw = m.get("home_team") or "Unknown"
        away_raw = m.get("away_team") or "Unknown"
        
        home = normalize(home_raw)
        away = normalize(away_raw)

        r_home = ratings.get(home)
        r_away = ratings.get(away)

        if r_home is None or r_away is None:
            continue # Skip teams the Elo model doesn't know

        p_h, p_d, p_a = predict_1x2(r_home, r_away)
        
        # Determine Top Pick (strictly from 1X2)
        picks = [("1", p_h), ("X", p_d), ("2", p_a)]
        best_pick, best_prob = max(picks, key=lambda x: x[1])

        match_str = f"{home_raw[:14]} vs {away_raw[:14]}"
        print(f"{match_str:<30} | {p_h:>8.1%}   | {p_d:>8.1%}   | {p_a:>8.1%}   | {best_pick} ({best_prob:.1%})")

    print("=" * 80)
    print(" ✅ COMPLETE. Use the 'Top Pick' column for your safest bet.")
    print("=" * 80)