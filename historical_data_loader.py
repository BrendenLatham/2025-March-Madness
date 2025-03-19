# historical_data_loader.py - Loads and Processes Historical NCAA Tournament Data from ESPN
import requests
import json
import pandas as pd
import os
import time
from tqdm import tqdm

class HistoricalTournamentDataLoader:
    """Utility for collecting and loading historical NCAA tournament data from ESPN."""
    
    def __init__(self, data_dir="historical_data"):
        self.data_dir = data_dir
        self.tournament_data = {}
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_historical_data(self, start_year=2010, end_year=2023):
        """Fetch historical NCAA tournament data from ESPN API."""
        print(f"Fetching historical tournament data from {start_year} to {end_year}...")
        
        for year in tqdm(range(start_year, end_year + 1)):
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={year}0301-{year}0405"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                self.extract_tournament_data(year, response.json())
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {year}: {e}")
            time.sleep(1)
        
        self.save_tournament_data()
    
    def extract_tournament_data(self, year, data):
        """Extracts relevant tournament information from ESPN API response."""
        games = data.get("events", [])
        
        tournament_results = {
            "year": year,
            "rounds": {},
            "champion": None
        }
        
        for game in games:
            round_name = game.get("season", {}).get("type", "Unknown")
            competitors = game.get("competitions", [{}])[0].get("competitors", [])
            
            if len(competitors) == 2:
                winner = next((team for team in competitors if team.get("winner")), None)
                loser = next((team for team in competitors if not team.get("winner")), None)
                
                if winner and loser:
                    if round_name not in tournament_results["rounds"]:
                        tournament_results["rounds"][round_name] = []
                    
                    tournament_results["rounds"][round_name].append({
                        "winner": winner["team"]["displayName"],
                        "loser": loser["team"]["displayName"],
                        "round": round_name
                    })
                    
                    if round_name == "Championship":
                        tournament_results["champion"] = winner["team"]["displayName"]
        
        self.tournament_data[str(year)] = tournament_results
    
    def save_tournament_data(self):
        """Save fetched tournament data to a JSON file."""
        file_path = os.path.join(self.data_dir, "tournaments.json")
        with open(file_path, 'w') as f:
            json.dump(self.tournament_data, f, indent=4)
        print(f"Saved tournament data to {file_path}")
    
    def load_tournament_data(self):
        """Load tournament data from saved JSON file."""
        file_path = os.path.join(self.data_dir, "tournaments.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.tournament_data = json.load(f)
            print(f"Loaded historical tournament data from {file_path}")
        else:
            print("No saved tournament data found. Fetching new data...")
            self.fetch_historical_data()

if __name__ == "__main__":
    loader = HistoricalTournamentDataLoader()
    loader.fetch_historical_data(start_year=2010, end_year=2023)
