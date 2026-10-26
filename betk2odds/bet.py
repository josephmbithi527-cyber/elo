import pandas as pd
import numpy as np
import requests
import io
import subprocess
import json
import re
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

# =============================================================================
# PART 0: FETCH MIDWEEK JACKPOT MATCHES FROM BETIKA
# =============================================================================
def fetch_jackpot_matches():
    print("🕷️ Fetching Midweek Jackpot matches from Betika via smartBetika...")
    
    try:
        # We call the smartBetika CLI. 
        result = subprocess.run(
            ["betika", "--midjp", "--get-API"], 
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0 and result.stdout:
            try:
                data = json.loads(result.stdout)
                matches = parse_jackpot_json(data)
                if matches:
                    print(f"✅ Successfully fetched {len(matches)} Midweek Jackpot matches.")
                    return matches
            except json.JSONDecodeError:
                # If it's not JSON, try parsing the text output
                matches = parse_jackpot_text(result.stdout)
                if matches:
                    print(f"✅ Successfully parsed {len(matches)} matches from text output.")
                    return matches
        
        # If the code reaches here, it means smartBetika ran but found NO matches
        print("❌ Betika API returned 0 matches. There is likely NO active Midweek Jackpot right now.")
        return []

    except FileNotFoundError:
        print("❌ ERROR: The 'smartBetika' library is not installed or not in your PATH.")
        print("Please run: pip install git+https://github.com/Simatwa/smartBetika.git")
        return []
    except Exception as e:
        print(f"❌ ERROR: Failed to run smartBetika. Details: {e}")
        return []

def parse_jackpot_json(data):
    matches = []
    try:
        for item in data:
            home = item.get('home_team') or item.get('home')
            away = item.get('away_team') or item.get('away')
            if home and away:
                matches.append({
                    "home": home, "away": away, "league": "Jackpot"
                })
    except Exception:
        pass
    return matches

def parse_jackpot_text(text):
    matches = []
    lines = text.split('\n')
    for line in lines:
        match = re.search(r'([A-Za-z\s\.]+)\s+vs\s+([A-Za-z\s\.]+)', line)
        if match:
            home = match.group(1).strip()
            away = match.group(2).strip()
            matches.append({
                "home": home, "away": away, "league": "Jackpot"
            })
    return matches

# =============================================================================
# PART 1: DOWNLOAD HISTORICAL DATA (EXPANDED)
# =============================================================================
def fetch_historical_data(target_date):
    print(f"📥 Downloading historical data for model training...")
    urls = {
        "EPL": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "LaLiga": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "SerieA": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "Bundesliga": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "Ligue1": "https://www.football-data.co.uk/mmz4281/2425/F1.csv"
    }
    
    all_dfs = []
    for league_name, url in urls.items():
        try:
            response = requests.get(url)
            df = pd.read_csv(io.StringIO(response.text))
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
            df['League'] = league_name
            all_dfs.append(df)
        except Exception:
            pass
            
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')
    combined_df = combined_df.dropna(subset=['Date'])
    combined_df = combined_df[combined_df['Date'] < target_date]
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    return combined_df

# =============================================================================
# PART 2: FEATURE ENGINEERING
# =============================================================================
def create_features(df):
    print("🛠️ Engineering features...")
    df['Target'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    features = ['HomeGoalsAvg', 'AwayGoalsAvg', 'HomeConcededAvg', 'AwayConcededAvg', 'HomePointsAvg', 'AwayPointsAvg']
    for col in features: df[col] = np.nan

    team_history = {}
    for index, row in df.iterrows():
        home, away = str(row['HomeTeam']), str(row['AwayTeam'])
        if home not in team_history: team_history[home] = {'scored': [1.5], 'conceded': [1.5], 'points': [1.5]}
        if away not in team_history: team_history[away] = {'scored': [1.5], 'conceded': [1.5], 'points': [1.5]}
            
        df.at[index, 'HomeGoalsAvg'] = np.mean(team_history[home]['scored'][-5:])
        df.at[index, 'AwayGoalsAvg'] = np.mean(team_history[away]['scored'][-5:])
        df.at[index, 'HomeConcededAvg'] = np.mean(team_history[home]['conceded'][-5:])
        df.at[index, 'AwayConcededAvg'] = np.mean(team_history[away]['conceded'][-5:])
        df.at[index, 'HomePointsAvg'] = np.mean(team_history[home]['points'][-5:])
        df.at[index, 'AwayPointsAvg'] = np.mean(team_history[away]['points'][-5:])
        
        h_goals, a_goals = row['FTHG'], row['FTAG']
        if h_goals > a_goals: h_pts, a_pts = 3, 0
        elif h_goals < a_goals: h_pts, a_pts = 0, 3
        else: h_pts, a_pts = 1, 1
            
        team_history[home]['scored'].append(h_goals)
        team_history[home]['conceded'].append(a_goals)
        team_history[home]['points'].append(h_pts)
        team_history[away]['scored'].append(a_goals)
        team_history[away]['conceded'].append(h_goals)
        team_history[away]['points'].append(a_pts)
    return df

# =============================================================================
# PART 3: TRAIN CATBOOST MODEL
# =============================================================================
def train_model(df):
    print("🧠 Training CatBoost Model...")
    cat_features = ['HomeTeam', 'AwayTeam', 'League']
    num_features = ['HomeGoalsAvg', 'AwayGoalsAvg', 'HomeConcededAvg', 'AwayConcededAvg', 'HomePointsAvg', 'AwayPointsAvg']
    features = cat_features + num_features
    for col in cat_features: df[col] = df[col].astype(str)
    
    X, y = df[features], df['Target']
    split_idx = int(len(df) * 0.9)
    
    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, loss_function='MultiClass', verbose=0, random_seed=42)
    model.fit(Pool(X.iloc[:split_idx], y.iloc[:split_idx], cat_features=cat_features), eval_set=Pool(X.iloc[split_idx:], y.iloc[split_idx:], cat_features=cat_features), early_stopping_rounds=30)
    
    print(f"✅ Model Trained! Accuracy: {accuracy_score(y.iloc[split_idx:], model.predict(X.iloc[split_idx:])) * 100:.2f}%")
    return model, features

# =============================================================================
# PART 4: PREDICT JACKPOT MATCHES
# =============================================================================
def predict_jackpot(model, df, features, jackpot_games):
    latest_stats = {}
    all_teams = set(df['HomeTeam'].unique()).union(set(df['AwayTeam'].unique()))
    
    for team in all_teams:
        team_df = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        if len(team_df) > 0:
            last_match = team_df.iloc[-1]
            if last_match['HomeTeam'] == team:
                latest_stats[team] = {'scored': last_match['HomeGoalsAvg'], 'conceded': last_match['HomeConcededAvg'], 'points': last_match['HomePointsAvg']}
            else:
                latest_stats[team] = {'scored': last_match['AwayGoalsAvg'], 'conceded': last_match['AwayConcededAvg'], 'points': last_match['AwayPointsAvg']}

    analyzed_games = []
    name_mapping = {
        "Atletico Madrid": "Ath Madrid", "Athletic Bilbao": "Ath Bilbao", "AC Milan": "Milan",
        "Inter Milan": "Inter", "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
        "Manchester City": "Man City", "Manchester United": "Man Utd", "Nottingham Forest": "Nott'm Forest",
        "Paris Saint Germain": "Paris SG", "PSG": "Paris SG", "Bayern Munich": "Bayern Munich",
        "Eintracht Frankfurt": "Frankfurt", "Borussia Mönchengladbach": "M'gladbach"
    }

    for game in jackpot_games:
        home_raw, away_raw = str(game['home']), str(game['away'])
        home = name_mapping.get(home_raw, home_raw)
        away = name_mapping.get(away_raw, away_raw)
        
        if home not in latest_stats or away not in latest_stats:
            print(f"⚠️ Skipping {home_raw} vs {away_raw}: Model has no historical data.")
            continue
            
        match_features = pd.DataFrame([{
            'HomeTeam': home, 'AwayTeam': away, 'League': 'Jackpot',
            'HomeGoalsAvg': latest_stats[home]['scored'], 'AwayGoalsAvg': latest_stats[away]['scored'],
            'HomeConcededAvg': latest_stats[home]['conceded'], 'AwayConcededAvg': latest_stats[away]['conceded'],
            'HomePointsAvg': latest_stats[home]['points'], 'AwayPointsAvg': latest_stats[away]['points'],
        }])
        
        probs = model.predict_proba(match_features)[0] 
        
        analyzed_games.append({
            "match": f"{home_raw} vs {away_raw}", 
            "home_prob": probs[0],
            "draw_prob": probs[1],
            "away_prob": probs[2],
            "pick": ['1 (Home)', 'X (Draw)', '2 (Away)'][np.argmax(probs)]
        })
    return analyzed_games

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    TARGET_DATE = pd.to_datetime("today").normalize() 
    
    # 1. Fetch Jackpot matches FIRST
    jackpot_matches = fetch_jackpot_matches()
    
    # 🛑 STRICT CHECK: Stop if there is no jackpot right now
    if not jackpot_matches:
        print("\n🛑 STOPPING SCRIPT: No active Midweek Jackpot found on Betika.")
        print("Please try again when a new Midweek Jackpot is live.")
        exit()
    
    # 2. Train Model (Only runs if a Jackpot exists)
    raw_data = fetch_historical_data(TARGET_DATE)
    if raw_data.empty: exit("No historical data.")
    
    featured_data = create_features(raw_data)
    model, features = train_model(featured_data)
    
    # 3. Predict
    print(f"\n🔮 Analyzing {len(jackpot_matches)} Jackpot matches...")
    predictions = predict_jackpot(model, featured_data, features, jackpot_matches)
    
    # 4. Print Results
    print("\n" + "=" * 90)
    print(f"🏆 BETIKA MIDWEEK JACKPOT 1X2 PREDICTIONS")
    print("=" * 90)
    print(f"{'Match':<35} | {'1 (Home)':<10} | {'X (Draw)':<10} | {'2 (Away)':<10} | {'Model Pick'}")
    print("-" * 90)
    
    if not predictions:
        print("❌ No matches could be analyzed. Check team name mappings.")
    else:
        for game in predictions:
            print(f"{game['match']:<35} | {game['home_prob']*100:>6.1f}%   | {game['draw_prob']*100:>6.1f}%   | {game['away_prob']*100:>6.1f}%   | {game['pick']}")
            
    print("=" * 90)