import json
import pandas as pd
import random
from collections import defaultdict

class MarchMadnessSimulator:
    def __init__(self, use_real_data=True):
        self.UPSET_FACTOR = 0.20
        self.MOMENTUM_FACTOR = 0.08
        self.use_real_data = use_real_data
        self.team_strength = {}

        if self.use_real_data:
            print("✅ Loading real NCAA data...")
            self.load_data()
        else:
            print("⚠ WARNING: Using mock data (Not recommended).")
            self.create_mock_data()

    def load_data(self):
        """
        Loads real NCAA team data from ESPN API results.
        """
        try:
            self.teams_info = pd.read_csv('ncaa_teams.csv')
            if self.teams_info.empty:
                raise ValueError("⚠ ncaa_teams.csv is empty!")

            print(f"✅ Loaded {len(self.teams_info)} teams.")
            self.calculate_team_strength()
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            self.create_mock_data()

    def create_mock_data(self):
        """
        Prevents fake teams from appearing.
        """
        print("⚠ ERROR: Real team data missing! Ensure ncaa_teams.csv is generated correctly.")

    def get_tournament_teams(self):
        """
        Fetches only valid teams.
        """
        if self.teams_info is None or self.teams_info.empty:
            print("❌ ERROR: No valid team data found!")
            return {}

        return {team['id']: team['name'] for _, team in self.teams_info.iterrows()}
