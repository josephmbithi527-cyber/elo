import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import io

# =============================================================================
# PART 1: THE FOOTBALL ELO MODEL (CORE MATH)
# =============================================================================
class FootballElo:
    def __init__(self, k_factor=20, home_advantage=65, draw_margin=0.25):
        self.k = k_factor
        self.home_adv = home_advantage
        self.draw_margin = draw_margin
        self.ratings = {}

    def get_rating(self, team):
        # Default rating for teams not found in historical data
        return self.ratings.get(team, 1500.0)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, home_team, away_team, home_goals, away_goals):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        expected_home = self.expected_score(home_rating + self.home_adv, away_rating)
        expected_away = 1.0 - expected_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            gd_multiplier = 1.0
        elif goal_diff == 2:
            gd_multiplier = 1.5
        else:
            gd_multiplier = (11.0 + goal_diff) / 8.0

        self.ratings[home_team] = home_rating + (self.k * gd_multiplier) * (actual_home - expected_home)
        self.ratings[away_team] = away_rating + (self.k * gd_multiplier) * (actual_away - expected_away)

    def get_win_probabilities(self, home_team, away_team):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        expected_home = self.expected_score(home_rating + self.home_adv, away_rating)
        
        elo_diff = abs((home_rating + self.home_adv) - away_rating)
        draw_prob = self.draw_margin - (elo_diff / 2000.0)
        draw_prob = max(0.10, min(0.35, draw_prob))
        
        remaining_prob = 1.0 - draw_prob
        
        return {
            "Home": expected_home * remaining_prob,
            "Draw": draw_prob,
            "Away": (1.0 - expected_home) * remaining_prob
        }

# =============================================================================
# PART 2: AUTOMATED DATA FETCHING
# =============================================================================
def fetch_historical_data():
    """Automatically downloads real historical EPL data to train the model."""
    print("📥 Downloading historical match data from the internet...")
    # Using football-data.co.uk for free historical EPL data (2023/2024 season)
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # Keep only the columns we need and drop empty rows
        df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        print(f"✅ Successfully downloaded {len(df)} historical matches.")
        return df
    except Exception as e:
        print(f"❌ Failed to download historical data: {e}")
        return pd.DataFrame()

def fetch_betika_matches():
    """Attempts to scrape live upcoming matches from Betika."""
    print("🕸️ Fetching upcoming matches from Betika...")
    
    # Note: Betika has dynamic APIs. This uses a standard API endpoint approach.
    # If they block this request, it falls back to sample data.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Example API endpoint (Betika often changes these)
    api_url = "https://api.betika.com/v1/uo/matches?tab=upcoming&sub_type_id=1,186,340&sport_id=14&tag_id=&sort_id=2&page=1&limit=50"
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        data = response.json()
        
        matches = []
        if 'data' in data:
            for item in data['data']:
                # Extract team names and odds
                home_team = item.get('home_team', '')
                away_team = item.get('away_team', '')
                
                # Betika's JSON structure for odds can be tricky. 
                # We look for the 1X2 market odds.
                odds_home = 1.0
                odds_away = 1.0
                odds_draw = 1.0
                
                # (Simplified extraction, actual Betika API keys vary)
                try:
                    odds_home = float(item['odds'][0]['home']) if 'odds' in item else 1.0
                    odds_away = float(item['odds'][0]['away']) if 'odds' in item else 1.0
                    odds_draw = float(item['odds'][0]['draw']) if 'odds' in item else 1.0
                except:
                    pass
                
                if home_team and away_team:
                    matches.append({
                        "match": f"{home_team} vs {away_team}",
                        "home": home_team,
                        "away": away_team,
                        "odds_home": odds_home,
                        "odds_away": odds_away,
                        "odds_draw": odds_draw
                    })
        
        if matches:
            print(f"✅ Successfully fetched {len(matches)} matches from Betika.")
            return matches
        else:
            raise ValueError("No matches found in API response.")

    except Exception as e:
        print(f"⚠️ Betika scraping blocked or failed ({e}).")
        print("🔄 Using fallback sample data for demonstration...")
        return [
            {"match": "Man City vs Chelsea", "home": "Man City", "away": "Chelsea", "odds_home": 1.50, "odds_away": 5.50, "odds_draw": 4.00},
            {"match": "Arsenal vs Liverpool", "home": "Arsenal", "away": "Liverpool", "odds_home": 2.20, "odds_away": 3.10, "odds_draw": 3.40},
            {"match": "Man United vs Tottenham", "home": "Man United", "away": "Tottenham", "odds_home": 2.40, "odds_away": 2.70, "odds_draw": 3.20},
            {"match": "Newcastle vs Aston Villa", "home": "Newcastle", "away": "Aston Villa", "odds_home": 2.10, "odds_away": 3.30, "odds_draw": 3.50}
        ]

# =============================================================================
# PART 3: THE MAIN HUNTER LOGIC
# =============================================================================
def main():
    print("=" * 80)
    print("🏆 THE FULLY AUTOMATED TOP 3 WIN PROBABILITY HUNTER 🏆")
    print("=" * 80)

    # 1. Fetch and Train
    hist_df = fetch_historical_data()
    if hist_df.empty:
        print("❌ Cannot proceed without training data.")
        return

    print("\n🧠 Training Elo Model on historical data...")
    elo = FootballElo(k_factor=20, home_advantage=65)
    
    for _, row in hist_df.iterrows():
        elo.update_ratings(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'])
    print(f"✅ Training Complete. Rated {len(elo.ratings)} unique teams.")

    # 2. Fetch Live Games
    upcoming_games = fetch_betika_matches()

    # 3. Analyze and Calculate
    print("\n📊 Analyzing matches and calculating probabilities...")
    analyzed_games = []
    
    for game in upcoming_games:
        probs = elo.get_win_probabilities(game['home'], game['away'])
        
        # Determine the best pick
        best_pick = max(probs, key=probs.get)
        
        if best_pick == "Home":
            pick_odds = game['odds_home']
        elif best_pick == "Away":
            pick_odds = game['odds_away']
        else:
            pick_odds = game['odds_draw']
            
        analyzed_games.append({
            "match": game['match'],
            "pick": best_pick,
            "odds": pick_odds,
            "win_prob": probs[best_pick]
        })

    # 4. Sort and Get Top 3
    sorted_games = sorted(analyzed_games, key=lambda x: x['win_prob'], reverse=True)
    top_3 = sorted_games[:3]

    # 5. Print Results
    print("\n" + "=" * 80)
    print(f"{'Match':<40} | {'Pick':<5} | {'Odds':<6} | {'Win Prob'}")
    print("-" * 80)
    
    if not top_3:
        print("❌ Could not find any games.")
    else:
        for game in top_3:
            formatted_prob = f"{game['win_prob'] * 100:.1f}%"
            print(f"{game['match']:<40} | {game['pick']:<5} | {game['odds']:<6.2f} | {formatted_prob}")
            
    print("=" * 80)

if __name__ == "__main__":
    main()