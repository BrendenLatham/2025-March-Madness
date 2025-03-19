# backtesting_framework.py - Runs Backtesting Against Historical Data
import json
import pandas as pd
import numpy as np
import os

class MarchMadnessBacktester:
    def __init__(self, simulator, years):
        """Initialize the backtester with a simulator and the years to test."""
        self.simulator = simulator
        self.years = years
        self.historical_data = {}
        self.metrics = {}
    
    def load_historical_tournaments(self, data_dir="historical_data"):
        """Load historical tournament data from JSON files."""
        file_path = os.path.join(data_dir, "tournaments.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.historical_data = json.load(f)
            print(f"Loaded data for {len(self.historical_data)} tournaments")
        else:
            raise FileNotFoundError("Historical tournament data missing. Run data collection first.")
    
    def run_backtests(self, num_simulations=500):
        """Run backtesting against historical tournament data."""
        all_metrics = {}
        
        for year in self.years:
            if str(year) not in self.historical_data:
                print(f"No data available for {year}, skipping...")
                continue
            
            real_winner = self.historical_data[str(year)].get("champion", {}).get("team_name", None)
            if not real_winner:
                print(f"Skipping {year}: No champion data available.")
                continue
            
            predicted_winners = self.simulator.simulate_tournament(num_simulations)
            accuracy = predicted_winners.count(real_winner) / num_simulations
            
            all_metrics[year] = {"accuracy": accuracy}
        
        with open("backtesting_results.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        
        print("Backtesting completed. Results saved to backtesting_results.json")
        return all_metrics

if __name__ == "__main__":
    from simulator import MarchMadnessSimulator
    simulator = MarchMadnessSimulator(use_real_data=True)
    backtester = MarchMadnessBacktester(simulator, years=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    backtester.load_historical_tournaments()
    backtester.run_backtests(num_simulations=500)
