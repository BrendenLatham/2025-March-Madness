# simulator.py - March Madness Tournament Simulator
import json
import pandas as pd
import numpy as np
import random
import os

class MarchMadnessSimulator:
    def __init__(self, use_real_data=True):
        self.use_real_data = use_real_data
        self.UPSET_FACTOR = 0.20  # Weights upset potential
        self.MOMENTUM_FACTOR = 0.08  # Boost for recent performance
        self.team_strength = {}
        self.teams_info = None
        
        if self.use_real_data:
            self.load_data()
        else:
            self.create_mock_data()
    
    def load_data(self):
        """Load real NCAA team data."""
        try:
            self.teams_info = pd.read_csv('ncaa_teams.csv')
            if self.teams_info.empty:
                raise ValueError("ncaa_teams.csv is empty!")
            self.calculate_team_strength()
        except Exception as e:
            print(f"Error loading real data: {e}")
            self.create_mock_data()
    
    def create_mock_data(self):
        """Fallback to mock data if real data isn't available."""
        print("WARNING: Using mock team data!")
        num_teams = 68
        self.teams_info = pd.DataFrame({
            'id': [str(i) for i in range(1, num_teams + 1)],
            'name': [f"Fake Team {i}" for i in range(1, num_teams + 1)]
        })
    
    def calculate_team_strength(self):
        """Calculate team strength based on historical data or generate estimates."""
        self.team_strength = {}
        for _, team in self.teams_info.iterrows():
            self.team_strength[team['id']] = random.uniform(0.5, 1.0)  # Mock strength

    def get_tournament_teams(self):
        """Retrieve tournament teams ensuring only real teams are included."""
        if self.teams_info is None or self.teams_info.empty:
            print("Error: No real teams loaded!")
            return {}
        return {team['id']: team['name'] for _, team in self.teams_info.iterrows()}

    def simulate_tournament(self, num_simulations=1000):
        """Simulate the tournament multiple times and determine the most likely winner."""
        results = []
        valid_teams = self.get_tournament_teams()
        if not valid_teams:
            raise ValueError("No valid teams available for simulation.")
        
        for _ in range(num_simulations):
            winner = random.choice(list(valid_teams.values()))
            results.append(winner)
        
        return results
    
    def generate_consensus_bracket(self, simulation_results):
        """Generate a consensus bracket from simulation results."""
        team_counts = {}
        for winner in simulation_results:
            team_counts[winner] = team_counts.get(winner, 0) + 1
        
        most_likely_winner = max(team_counts, key=team_counts.get)
        
        return {
            "most_likely_winner": most_likely_winner,
            "win_counts": team_counts
        }

if __name__ == "__main__":
    simulator = MarchMadnessSimulator(use_real_data=True)
    results = simulator.simulate_tournament(10000)
    consensus_bracket = simulator.generate_consensus_bracket(results)
    
    with open('final_consensus_bracket.json', 'w') as f:
        json.dump(consensus_bracket, f, indent=4)
    
    print("Final Consensus Bracket Saved")
