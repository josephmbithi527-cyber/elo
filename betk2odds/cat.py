import pandas as pd
import numpy as np
import requests
import io
import time
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =============================================================================
# PART 1: HISTORICAL DATA FETCHING (For Training)
# =============================================================================
def fetch_historical_data(target_date):
    print(f"📥 Downloading historical data for model training...")
    
    urls = {
        "EPL": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "Championship": "https://www.football-data.co.uk/mmz4281/2425/E1.csv"
    }
    
    all_dfs = []
    for league_name, url in urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
            df['League'] = league_name
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not download {league_name}: {e}")
            
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')
    combined_df = combined_df.dropna(subset=['Date'])
    
    # Filter: Only use matches BEFORE today to prevent data leakage
    combined_df = combined_df[combined_df['Date'] < target_date]
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Loaded {len(combined_df)} historical matches for training.")
    return combined_df

# =============================================================================
# PART 2: LIVE BETIKA SCRAPER (Uses Selenium)
# =============================================================================
def fetch_betika_matches():
    print("🕸️ Opening Betika to scrape today's matches...")
    
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Uncomment to run invisibly
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get("https://www.betika.com/en-ke/")
        
        # Wait for matches to load (max 15 seconds)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".prebet-match"))
        )
        
        # Scroll down to load more matches if necessary
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
        match_rows = driver.find_elements(By.CSS_SELECTOR, ".prebet-match")
        upcoming_games = []
        
        for row in match_rows:
            try:
                # Extract Team Names
                home_team = row.find_element(By.CSS_SELECTOR, ".prebet-match__teams__home").text.strip()
                away_team = row.find_element(By.CSS_SELECTOR, ".prebet-match__teams__home + span").text.strip()
                
                # Extract Odds (1, X, 2)
                odds_elements = row.find_elements(By.CSS_SELECTOR, ".prebet-match__odd .bold")
                if len(odds_elements) >= 3:
                    odds_home = float(odds_elements[0].text)
                    odds_draw = float(odds_elements[1].text)
                    odds_away = float(odds_elements[2].text)
                    
                    # Try to extract the league (if available in the row)
                    # Note: Depending on Betika's layout, this might need adjustment.
                    league_text = "Unknown"
                    
                    upcoming_games.append({
                        "home": home_team,
                        "away": away_team,
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                        "league": league_text
                    })
            except Exception:
                continue # Skip rows that are malformed or ads
                
        driver.quit()
        print(f"✅ Successfully scraped {len(upcoming_games)} matches from Betika.")
        return upcoming_games

    except Exception as e:
        print(f"❌ Failed to scrape Betika: {e}")
        driver.quit()
        return []

# =============================================================================
# PART 3: FEATURE ENGINEERING
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
            
        team_history[home]['scored'].append(h_goals); team_history[home]['conceded'].append(a_goals); team_history[home]['points'].append(h_pts)
        team_history[away]['scored'].append(a_goals); team_history[away]['conceded'].append(h_goals); team_history[away]['points'].append(a_pts)
    return df

# =============================================================================
# PART 4: MODEL TRAINING
# =============================================================================
def train_model(df):
    print("🧠 Training CatBoost Model...")
    cat_features = ['HomeTeam', 'AwayTeam', 'League']
    num_features = ['HomeGoalsAvg', 'AwayGoalsAvg', 'HomeConcededAvg', 'AwayConcededAvg', 'HomePointsAvg', 'AwayPointsAvg']
    features = cat_features + num_features
    
    for col in cat_features: df[col] = df[col].astype(str)
    X, y = df[features], df['Target']
    
    split_idx = int(len(df) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, loss_function='MultiClass', verbose=0, random_seed=42)
    model.fit(Pool(X_train, y_train, cat_features=cat_features), eval_set=Pool(X_test, y_test, cat_features=cat_features), early_stopping_rounds=30)
    print(f"✅ Model Trained! Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
    return model, features

# =============================================================================
# PART 5: PREDICTING SCRAPED MATCHES
# =============================================================================
def predict_betika_matches(model, df, features, betika_games):
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
    for game in betika_games:
        home, away = str(game['home']), str(game['away'])
        
        # Check if model knows these teams (Betika has global leagues, model only knows EPL/Champ)
        if home not in latest_stats or away not in latest_stats:
            # Optional: Print skipping message for debugging
            # print(f"⚠️ Skipping {home} vs {away}: No historical data in model.")
            continue
            
        match_features = pd.DataFrame([{
            'HomeTeam': home, 'AwayTeam': away, 'League': str(game.get('league', 'Unknown')),
            'HomeGoalsAvg': latest_stats[home]['scored'], 'AwayGoalsAvg': latest_stats[away]['scored'],
            'HomeConcededAvg': latest_stats[home]['conceded'], 'AwayConcededAvg': latest_stats[away]['conceded'],
            'HomePointsAvg': latest_stats[home]['points'], 'AwayPointsAvg': latest_stats[away]['points'],
        }])
        
        probs = model.predict_proba(match_features)[0]
        outcomes = ['Home', 'Draw', 'Away']
        best_pick_idx = np.argmax(probs)
        
        analyzed_games.append({
            "match": f"{home} vs {away}",
            "pick": outcomes[best_pick_idx],
            "odds": game['odds_home'] if outcomes[best_pick_idx] == 'Home' else (game['odds_away'] if outcomes[best_pick_idx] == 'Away' else game['odds_draw']),
            "win_prob": probs[best_pick_idx]
        })
    return analyzed_games

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    TARGET_DATE = pd.to_datetime("today").normalize() 
    
    # 1. Train Model on Historical Data
    raw_historical = fetch_historical_data(TARGET_DATE)
    if raw_historical.empty:
        print("❌ No historical data found. Exiting.")
        exit()
        
    featured_data = create_features(raw_historical)
    model, features = train_model(featured_data)
    
    # 2. Fetch Today's Matches from Betika
    todays_matches = fetch_betika_matches()
    
    if not todays_matches:
        print("❌ Could not fetch matches from Betika. The site might be blocking automated requests.")
        exit()

    # 3. Predict
    predictions = predict_betika_matches(model, featured_data, features, todays_matches)
    
    if not predictions:
        print("⚠️ None of the matches scraped from Betika were found in the model's training data (EPL/Championship).")
        print("The model only knows English teams. Betika may be showing other leagues today.")
        exit()

    # 4. Sort and Display
    predictions = sorted(predictions, key=lambda x: x['win_prob'], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"🎯 Top Predictions from Betika for TODAY ({TARGET_DATE.date()})")
    print(f"{'Match':<40} | {'Pick':<5} | {'Odds':<6} | {'Win Prob'}")
    print("-" * 80)
    for game in predictions[:5]: # Show top 5
        print(f"{game['match']:<40} | {game['pick']:<5} | {game['odds']:<6.2f} | {game['win_prob'] * 100:.1f}%")
    print("=" * 80)