import requests
import json
import pandas as pd
import os
import time
from tqdm import tqdm
from bs4 import BeautifulSoup

class HistoricalTournamentDataLoader:
    """
    Utility for collecting historical NCAA tournament data for backtesting.
    """
    
    def __init__(self, start_year=2010, end_year=2023, data_dir="historical_data"):
        """
        Initialize the data loader.
        
        Args:
            start_year: First year to collect data for
            end_year: Last year to collect data for
            data_dir: Directory to store the data
        """
        self.start_year = start_year
        self.end_year = end_year
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Store for collected data
        self.tournament_data = {}
    
    def collect_historical_data(self):
        """Collect historical NCAA tournament data for all years in range."""
        print(f"Collecting historical tournament data for {self.start_year}-{self.end_year}...")
        
        for year in range(self.start_year, self.end_year + 1):
            print(f"Collecting data for {year}...")
            try:
                self.collect_data_for_year(year)
                # Add a small delay to be nice to the servers
                time.sleep(1)
            except Exception as e:
                print(f"Error collecting data for {year}: {e}")
        
        # Save the compiled data
        self.save_tournament_data()
    
    def collect_data_for_year(self, year):
        """
        Collect data for a specific tournament year.
        
        Args:
            year: The year to collect data for
        """
        # Initialize data structure for this year
        self.tournament_data[str(year)] = {
            "regions": [],
            "final_four": [],
            "champion": None,
            "rounds": {
                "round_1": [],
                "round_2": [],
                "round_3": [],
                "round_4": [],
                "round_5": [],
                "round_6": []
            },
            "upsets": []
        }
        
        # Collect tournament bracket data
        self._collect_bracket_data(year)
        
        # Collect team statistics for that year (if needed)
        # self._collect_team_statistics(year)
        
        print(f"Completed data collection for {year}")
    
    def _collect_bracket_data(self, year):
        """
        Collect bracket and results data for a specific year.
        
        Args:
            year: The year to collect data for
        """
        # Note: In a real implementation, you would scrape from sports sites
        # or use an API to get historical bracket data.
        # For demonstration, we'll create simulated data.
        
        # For recent years, we can try to get real data
        if year >= 2015:
            success = self._try_fetch_real_bracket_data(year)
            if success:
                return
        
        # Fall back to simulated data if needed
        self._generate_simulated_bracket_data(year)
    
    def _try_fetch_real_bracket_data(self, year):
        """
        Attempt to fetch real bracket data for a year.
        
        Args:
            year: The year to fetch data for
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Could use sports-reference.com, ESPN, or another source
        # For demonstration, we'll show how to extract from an API or web page
        
        # Example API attempt (note: this is a placeholder URL)
        api_url = f"https://api.example.com/ncaa/tournament/{year}"
        
        try:
            # response = requests.get(api_url)
            # if response.status_code == 200:
            #     data = response.json()
            #     # Process the data...
            #     return True
            
            # If API fails, try web scraping a results page
            # results_url = f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html"
            # html = requests.get(results_url).text
            # self._parse_results_page(html, year)
            # return True
            
            # For now, return False to fall back to simulated data
            return False
            
        except Exception as e:
            print(f"Error fetching real data for {year}: {e}")
            return False
    
    def _parse_results_page(self, html, year):
        """
        Parse HTML from a results page to extract tournament data.
        
        Args:
            html: The HTML content
            year: The tournament year
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract regions
            regions = []
            region_elements = soup.select('.region-name')
            for element in region_elements:
                regions.append(element.text.strip())
            
            self.tournament_data[str(year)]["regions"] = regions
            
            # Extract Final Four teams
            ff_elements = soup.select('.final-four-team')
            for element in ff_elements:
                team_name = element.select_one('.team-name').text.strip()
                seed = int(element.select_one('.seed').text.strip())
                region = element.select_one('.region').text.strip()
                
                self.tournament_data[str(year)]["final_four"].append({
                    "team_name": team_name,
                    "seed": seed,
                    "region": region
                })
            
            # Extract champion
            champion_element = soup.select_one('.champion-team')
            if champion_element:
                team_name = champion_element.select_one('.team-name').text.strip()
                seed = int(champion_element.select_one('.seed').text.strip())
                region = champion_element.select_one('.region').text.strip()
                
                self.tournament_data[str(year)]["champion"] = {
                    "team_name": team_name,
                    "seed": seed,
                    "region": region
                }
            
            # Extract round results and upsets
            # This would be more complex and depends on the page structure
        except Exception as e:
            print(f"Error parsing results page for {year}: {e}")
    
    def _generate_simulated_bracket_data(self, year):
        """
        Generate simulated bracket data for a specific year.
        
        Args:
            year: The year to generate data for
        """
        import random
        
        # Regions for the tournament
        regions = ["East", "West", "South", "Midwest"]
        self.tournament_data[str(year)]["regions"] = regions
        
        # Create a 64-team bracket (simplified, without play-in games)
        all_teams = []
        
        # Generate teams for each region with seeds 1-16
        for region in regions:
            for seed in range(1, 17):
                team = {
                    "team_name": f"Team {seed} ({region})",
                    "seed": seed,
                    "region": region
                }
                all_teams.append(team)
        
        # Simulate tournament rounds
        # First round - use historically realistic probabilities
        first_round_matchups = [
            (1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)
        ]
        
        first_round_winners = []
        upsets = []
        
        for region in regions:
            for seed1, seed2 in first_round_matchups:
                # Probabilities based on historical NCAA tournament data
                upset_probs = {
                    (1, 16): 0.01,   # 1 seed vs 16 seed
                    (2, 15): 0.06,   # 2 seed vs 15 seed
                    (3, 14): 0.15,   # etc.
                    (4, 13): 0.20,
                    (5, 12): 0.35,   # The famous "12-5 upset"
                    (6, 11): 0.40,
                    (7, 10): 0.40,
                    (8, 9): 0.50,    # 8-9 is basically 50/50
                }
                
                # Determine the winner
                if random.random() < upset_probs.get((seed1, seed2), 0.5):
                    # Lower seed wins (upset if not 8-9 matchup)
                    winner_seed = seed2
                    if seed1 < seed2 and (seed1, seed2) != (8, 9):
                        upsets.append([seed2, seed1, 1, region])  # [winner_seed, loser_seed, round, region]
                else:
                    # Higher seed wins
                    winner_seed = seed1
                
                # Record the result
                self.tournament_data[str(year)]["rounds"]["round_1"].append([winner_seed, 
                                                                          (seed2 if winner_seed == seed1 else seed1), 
                                                                          region])
                
                # Track the winner for next round
                winner = next((team for team in all_teams if team["seed"] == winner_seed and team["region"] == region), None)
                if winner:
                    first_round_winners.append(winner)
        
        # Simulate later rounds (simplified)
        # In a real implementation, you would follow proper bracket progression
        second_round_winners = self._simulate_round(first_round_winners, 2, upsets)
        sweet_16_winners = self._simulate_round(second_round_winners, 3, upsets)
        elite_8_winners = self._simulate_round(sweet_16_winners, 4, upsets)
        
        # Final Four - winners from each region
        self.tournament_data[str(year)]["final_four"] = elite_8_winners
        
        # Championship game and champion
        if len(elite_8_winners) >= 2:
            championship_teams = elite_8_winners.copy()
            random.shuffle(championship_teams)
            
            # Simulate championship with bias toward lower seeds
            team1, team2 = championship_teams[0], championship_teams[1]
            
            # Lower seed has advantage
            if team1["seed"] < team2["seed"]:
                champion = team1 if random.random() < 0.7 else team2
            else:
                champion = team2 if random.random() < 0.7 else team1
            
            self.tournament_data[str(year)]["champion"] = champion
            
            # Record championship game
            self.tournament_data[str(year)]["rounds"]["round_6"].append([
                champion["seed"],
                team2["seed"] if champion["seed"] == team1["seed"] else team1["seed"],
                "Championship"
            ])
        
        # Record upsets
        self.tournament_data[str(year)]["upsets"] = upsets
    
    def _simulate_round(self, previous_round_winners, round_number, upsets):
        """
        Simulate a tournament round.
        
        Args:
            previous_round_winners: Winners from the previous round
            round_number: The current round number (2-5)
            upsets: List to track upsets
            
        Returns:
            List of winners for this round
        """
        import random
        
        winners = []
        matchups = []
        
        # Create matchups based on bracket structure (simplified)
        remaining_teams = previous_round_winners.copy()
        random.shuffle(remaining_teams)
        
        for i in range(0, len(remaining_teams), 2):
            if i + 1 < len(remaining_teams):
                matchups.append((remaining_teams[i], remaining_teams[i+1]))
        
        # Determine winners for each matchup
        for team1, team2 in matchups:
            # Higher seed (lower number) has advantage
            if team1["seed"] < team2["seed"]:
                # team1 is the favorite - 70% chance to win
                winner = team1 if random.random() < 0.7 else team2
                loser = team2 if winner == team1 else team1
            else:
                # team2 is the favorite - 70% chance to win
                winner = team2 if random.random() < 0.7 else team1
                loser = team1 if winner == team2 else team1
            
            # Check for upset
            if winner["seed"] > loser["seed"]:
                upsets.append([winner["seed"], loser["seed"], round_number, winner["region"]])
            
            # Record the result
            self.tournament_data[str(year)]["rounds"][f"round_{round_number}"].append([
                winner["seed"], loser["seed"], winner["region"]
            ])
            
            winners.append(winner)
        
        return winners
    
    def _collect_team_statistics(self, year):
        """
        Collect team statistics for a specific year.
        
        Args:
            year: The year to collect stats for
        """
        # This would fetch team stats like KenPom ratings, offensive/defensive efficiency, etc.
        # Useful for training advanced metric models
        
        # For now, we'll skip this as it's not immediately needed for backtesting
        pass
    
    def save_tournament_data(self):
        """Save the collected tournament data to a JSON file."""
        output_file = os.path.join(self.data_dir, "tournaments.json")
        
        print(f"Saving tournament data to {output_file}...")
        
        with open(output_file, 'w') as f:
            json.dump(self.tournament_data, f, indent=2)
        
        print(f"Saved data for {len(self.tournament_data)} tournaments")
    
    def load_tournament_data(self):
        """Load tournament data from the JSON file."""
        input_file = os.path.join(self.data_dir, "tournaments.json")
        
        if not os.path.exists(input_file):
            print(f"No tournament data file found at {input_file}")
            return
        
        print(f"Loading tournament data from {input_file}...")
        
        with open(input_file, 'r') as f:
            self.tournament_data = json.load(f)
        
        print(f"Loaded data for {len(self.tournament_data)} tournaments")
    
    def get_historical_upsets(self):
        """
        Analyze historical upset patterns.
        
        Returns:
            DataFrame with upset statistics
        """
        all_upsets = []
        
        for year, data in self.tournament_data.items():
            if "upsets" in data:
                for upset in data["upsets"]:
                    all_upsets.append({
                        "year": year,
                        "winner_seed": upset[0],
                        "loser_seed": upset[1],
                        "round": upset[2],
                        "region": upset[3]
                    })
        
        if not all_upsets:
            print("No upset data available")
            return None
        
        upsets_df = pd.DataFrame(all_upsets)
        
        # Calculate upset frequency by matchup
        matchup_counts = upsets_df.groupby(["winner_seed", "loser_seed"]).size().reset_index(name="count")
        
        # Calculate upset frequency by round
        round_counts = upsets_df.groupby("round").size().reset_index(name="count")
        
        print(f"Found {len(all_upsets)} historical upsets")
        print("\nUpset frequency by matchup:")
        print(matchup_counts.sort_values("count", ascending=False).head(10))
        print("\nUpset frequency by round:")
        print(round_counts.sort_values("round"))
        
        return upsets_df
    
    def get_seed_performance(self):
        """
        Analyze how different seeds perform historically.
        
        Returns:
            DataFrame with seed performance statistics
        """
        seed_stats = {i: {"final_four": 0, "elite_eight": 0, "sweet_sixteen": 0, "championships": 0} 
                     for i in range(1, 17)}
        
        for year, data in self.tournament_data.items():
            # Count Final Four appearances
            for team in data.get("final_four", []):
                seed = team.get("seed")
                if seed and 1 <= seed <= 16:
                    seed_stats[seed]["final_four"] += 1
            
            # Count championships
            if data.get("champion"):
                seed = data["champion"].get("seed")
                if seed and 1 <= seed <= 16:
                    seed_stats[seed]["championships"] += 1
            
            # Count Elite Eight appearances
            for result in data.get("rounds", {}).get("round_4", []):
                if len(result) >= 2:
                    seed = result[0]  # Winner seed
                    if 1 <= seed <= 16:
                        seed_stats[seed]["elite_eight"] += 1
            
            # Count Sweet Sixteen appearances
            for result in data.get("rounds", {}).get("round_3", []):
                if len(result) >= 2:
                    seed = result[0]  # Winner seed
                    if 1 <= seed <= 16:
                        seed_stats[seed]["sweet_sixteen"] += 1
        
        # Convert to DataFrame
        stats_list = []
        for seed, stats in seed_stats.items():
            stats_list.append({
                "seed": seed,
                "final_four": stats["final_four"],
                "elite_eight": stats["elite_eight"],
                "sweet_sixteen": stats["sweet_sixteen"],
                "championships": stats["championships"]
            })
        
        seed_df = pd.DataFrame(stats_list)
        seed_df = seed_df.sort_values("seed")
        
        print("Historical seed performance:")
        print(seed_df)
        
        return seed_df
