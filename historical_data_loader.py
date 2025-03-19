import requests
import json
import os
import time
from tqdm import tqdm

class HistoricalTournamentDataLoader:
    """
    Fetches historical NCAA tournament data from ESPN API.
    """

    def __init__(self, start_year=2010, end_year=2023, data_dir="historical_data"):
        self.start_year = start_year
        self.end_year = end_year
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.tournament_data = {}

    def fetch_tournament_results(self, year):
        """
        Fetches tournament results for a given year from ESPN.
        """
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={year}0301-{year}0405"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"⚠ Warning: Could not fetch tournament data for {year}. Status Code: {response.status_code}")
            return None

        data = response.json()
        games = data.get("events", [])

        tournament_results = {"rounds": {}}

        for game in games:
            round_name = game.get("season", {}).get("type", "Unknown")
            competitors = game.get("competitions", [{}])[0].get("competitors", [])

            if len(competitors) == 2:
                winner = [team for team in competitors if team["winner"]][0]
                loser = [team for team in competitors if not team["winner"]][0]

                if round_name not in tournament_results["rounds"]:
                    tournament_results["rounds"][round_name] = []

                tournament_results["rounds"][round_name].append({
                    "winner": winner["team"]["displayName"],
                    "loser": loser["team"]["displayName"],
                    "round": round_name
                })

        return tournament_results

    def collect_historical_data(self):
        """
        Fetches historical data for all years.
        """
        print(f"Fetching historical tournament data from {self.start_year} to {self.end_year}...")

        for year in tqdm(range(self.start_year, self.end_year + 1), desc="Fetching tournament data"):
            results = self.fetch_tournament_results(year)
            if results:
                self.tournament_data[str(year)] = results
            time.sleep(1)  # Rate limiting

        with open(f"{self.data_dir}/tournaments.json", "w") as f:
            json.dump(self.tournament_data, f, indent=2)

        print("✅ Tournament data saved.")

if __name__ == "__main__":
    loader = HistoricalTournamentDataLoader()
    loader.collect_historical_data()
