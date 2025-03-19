# data_collection.py - Fetch NCAA Data from API
import pandas as pd
import requests
import json
import time
import os

def get_ncaa_teams():
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=400"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    teams = []
    for item in data['sports'][0]['leagues'][0]['teams']:
        team = item['team']
        teams.append({
            'id': team['id'],
            'name': team['displayName'],
            'abbreviation': team.get('abbreviation', ''),
            'conference': team.get('conferenceId', '')
        })
    
    df = pd.DataFrame(teams)
    df.to_csv('ncaa_teams.csv', index=False)
    return df

def get_historical_brackets(start_year=2010, end_year=2023):
    all_tournaments = {}
    
    for year in range(start_year, end_year + 1):
        url = f"https://api.sportsdata.io/v3/cbb/scores/json/Tournament/{year}"
        headers = {"Ocp-Apim-Subscription-Key": os.getenv("SPORTSDATA_API_KEY")}  # Ensure API key is stored securely
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            all_tournaments[year] = response.json()
        else:
            print(f"Warning: Could not fetch tournament data for {year}. Status Code: {response.status_code}")
        time.sleep(1)
    
    with open('historical_brackets.json', 'w') as f:
        json.dump(all_tournaments, f, indent=4)
    return all_tournaments

# Run data collection
if __name__ == "__main__":
    get_ncaa_teams()
    get_historical_brackets()


# simulator.py - Simulates March Madness Tournament
import pandas as pd
import random

class MarchMadnessSimulator:
    def __init__(self, use_real_data=True):
        self.use_real_data = use_real_data
        self.teams_info = None
        self.team_strength = {}
        
        if self.use_real_data:
            self.load_data()
        else:
            self.create_mock_data()
    
    def load_data(self):
        try:
            self.teams_info = pd.read_csv('ncaa_teams.csv')
            if self.teams_info.empty:
                raise ValueError("ncaa_teams.csv is empty!")
            self.calculate_team_strength()
        except Exception as e:
            print(f"Error loading real data: {e}")
            self.create_mock_data()
    
    def create_mock_data(self):
        num_teams = 68
        self.teams_info = pd.DataFrame({
            'id': [str(i) for i in range(1, num_teams + 1)],
            'name': [f"Fake Team {i}" for i in range(1, num_teams + 1)]
        })
    
    def calculate_team_strength(self):
        self.team_strength = {team['id']: random.uniform(0.5, 1.0) for _, team in self.teams_info.iterrows()}
    
    def get_tournament_teams(self):
        if self.teams_info is None or self.teams_info.empty:
            print("Error: No real teams loaded!")
            return {}
        return {team['id']: team['name'] for _, team in self.teams_info.iterrows()}
    
    def simulate_tournament(self, num_simulations=1000):
        results = []
        valid_teams = self.get_tournament_teams()
        for _ in range(num_simulations):
            winner = random.choice(list(valid_teams.values()))
            results.append(winner)
        return results


# backtesting_framework.py - Runs Backtesting
import json

def run_backtesting(simulator, num_simulations=500):
    with open('historical_brackets.json', 'r') as f:
        historical_data = json.load(f)
    
    accuracy_results = {}
    for year, tournament in historical_data.items():
        predicted_winners = simulator.simulate_tournament(num_simulations)
        real_winner = tournament.get('champion', {}).get('name', None)
        if real_winner:
            accuracy_results[year] = predicted_winners.count(real_winner) / num_simulations
    
    with open('backtesting_results.json', 'w') as f:
        json.dump(accuracy_results, f, indent=4)
    
    return accuracy_results


# run_backtesting.py - Main Execution Script
if __name__ == "__main__":
    simulator = MarchMadnessSimulator(use_real_data=True)
    run_backtesting(simulator)
    final_predictions = simulator.simulate_tournament(10000)
    with open('final_consensus_bracket.json', 'w') as f:
        json.dump(final_predictions, f, indent=4)
    print("Final March Madness Predictions Saved")
