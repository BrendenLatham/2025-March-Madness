import json
import pandas as pd

class MarchMadnessBacktester:
    def __init__(self, simulator, years=None):
        self.simulator = simulator
        self.years = years or list(range(2010, 2023))
        self.historical_data = {}

    def load_historical_tournaments(self):
        """
        Loads ESPN tournament data.
        """
        try:
            with open("historical_data/tournaments.json", "r") as f:
                self.historical_data = json.load(f)
            print(f"✅ Loaded historical tournament data for {len(self.historical_data)} years.")
        except Exception as e:
            print(f"❌ ERROR: Could not load tournament data. {e}")

    def run_backtest(self):
        """
        Simulates tournaments and compares with historical ESPN results.
        """
        for year in self.years:
            if str(year) not in self.historical_data:
                print(f"⚠ No data for {year}, skipping...")
                continue

            print(f"🔄 Running backtest for {year}...")
            simulated_results = self.simulator.run_simulation()
            actual_results = self.historical_data[str(year)]["rounds"]

            # Compare simulated vs actual results
            correct_predictions = sum(
                1 for rnd in simulated_results["rounds"] if rnd in actual_results
            )

            print(f"✅ {correct_predictions} correct predictions for {year}.")

if __name__ == "__main__":
    from simulator import MarchMadnessSimulator

    simulator = MarchMadnessSimulator(use_real_data=True)
    backtester = MarchMadnessBacktester(simulator)
    backtester.load_historical_tournaments()
    backtester.run_backtest()
