# March Madness Tournament Simulator
import json
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class MarchMadnessSimulator:
    """
    A comprehensive simulator for March Madness tournament brackets
    that accounts for team strength, upset potential, and tournament dynamics.
    """

    def __init__(self, use_real_data=False):
        # Constants for simulation adjustments
        self.UPSET_FACTOR = 0.20  # How much to weight upset potential (higher = more upsets)
        self.HOME_ADVANTAGE = 0.05  # Small boost for teams playing closer to home
        self.MOMENTUM_FACTOR = 0.08  # Teams that win by larger margins get momentum
        self.CONFERENCE_STRENGTH = {
            # Power conferences get a slight boost
            'ACC': 0.03,
            'Big Ten': 0.03,
            'Big 12': 0.04,
            'Big East': 0.02,
            'SEC': 0.03,
            'Pac-12': 0.02,
            # Mid-majors that often perform well in March
            'West Coast': 0.01,
            'American': 0.01,
            'Mountain West': 0.01,
            # Default for other conferences
            'default': 0.0
        }

        # Historical upset rates by seed matchup
        self.HISTORICAL_UPSET_RATES = {
            (1, 16): 0.01,  # 1 vs 16 seeds - only happened once
            (2, 15): 0.06,  # 2 vs 15 seeds
            (3, 14): 0.13,  # 3 vs 14 seeds
            (4, 13): 0.21,  # 4 vs 13 seeds
            (5, 12): 0.35,  # 5 vs 12 seeds - the famous 12-5 upset
            (6, 11): 0.37,  # 6 vs 11 seeds
            (7, 10): 0.39,  # 7 vs 10 seeds
            (8, 9): 0.49,   # 8 vs 9 seeds - almost even matchup
        }

        # Load data
        if use_real_data:
            self.load_data()
        else:
            self.create_mock_data()

    def load_data(self):
        """Load required data files for the simulation"""
        try:
            # Load team statistics
            try:
                self.team_stats = pd.read_csv('team_stats_final.csv')
                print(f"Loaded team stats for {len(self.team_stats)} teams")
            except:
                print("Warning: Could not load team_stats_final.csv")
                self.team_stats = None

            # Load bracketology or current bracket information
            try:
                self.bracket_data = pd.read_csv('bracketology.csv')
                print(f"Loaded bracketology data with {len(self.bracket_data)} entries")
            except:
                print("Warning: Could not load bracketology.csv, will use rankings instead")
                self.bracket_data = None

                # Alternative: use rankings if bracketology not available
                try:
                    self.rankings = pd.read_csv('ncaa_rankings.csv')
                    print(f"Loaded rankings data for {len(self.rankings)} teams")
                except:
                    print("Warning: Could not load ncaa_rankings.csv")
                    self.rankings = None

            # Load team information
            try:
                self.teams_info = pd.read_csv('ncaa_teams.csv')
                print(f"Loaded info for {len(self.teams_info)} teams")
            except:
                print("Warning: Could not load ncaa_teams.csv")
                self.teams_info = None

            # Create team strength dictionary
            self.calculate_team_strength()

        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to mock data...")
            self.create_mock_data()

    def create_mock_data(self):
      print("Creating mock NCAA data for simulation...")

      # Create consistent data lengths
      num_teams = 68
      team_ids = [str(i) for i in range(1, num_teams+1)]
      team_names = [f"Team {i}" for i in range(1, num_teams+1)]

      # Create team info with matching array lengths
      self.teams_info = pd.DataFrame({
          'id': team_ids,
          'name': team_names,
          'abbreviation': [f"T{i}" for i in range(1, num_teams+1)],
          'location': [f"Location {i}" for i in range(1, num_teams+1)],
          'conference': (['ACC'] * 14 + ['Big Ten'] * 14 + ['Big 12'] * 14 +
                        ['SEC'] * 14 + ['Pac-12'] * 12)[:num_teams]  # Ensure this is exactly num_teams long
      })

      # Create rankings with matching array lengths
      self.rankings = pd.DataFrame({
          'poll': ["AP Top 25"] * num_teams,
          'rank': list(range(1, num_teams+1)),
          'team_id': team_ids,
          'team_name': team_names
      })

      # Create team strengths
      self.team_strength = {}
      for i in range(1, num_teams+1):
          # Base strength inversely related to rank (1-68)
          # Teams 1-10: 0.8-0.95, Teams 11-30: 0.65-0.8, Teams 31-50: 0.5-0.65, Teams 51-68: 0.35-0.5
          base = 0.95 - (i / num_teams) * 0.6
          # Add small random variation
          self.team_strength[str(i)] = max(0.1, min(base + random.uniform(-0.05, 0.05), 0.95))

    def calculate_team_strength(self):
        """
        Calculate an overall strength rating for each team based on statistics
        """
        self.team_strength = {}

        if self.team_stats is None or len(self.team_stats) == 0:
            print("No team statistics available. Using rankings to estimate team strength.")
            self._create_mock_strengths()
            return

        # Normalize and weight different statistical categories
        for _, team in self.team_stats.iterrows():
            team_id = str(team['team_id'])

            # Base strength factors
            strength = 0.0

            # Add offensive metrics (if they exist in the data)
            try:
                # Shooting efficiency
                if 'offensive_fieldGoalPct' in team:
                    strength += float(team['offensive_fieldGoalPct']) * 0.15

                # Scoring
                if 'offensive_avgPoints' in team:
                    # Normalize points (assuming max of 90 points per game)
                    normalized_points = min(float(team['offensive_avgPoints']) / 90.0, 1.0)
                    strength += normalized_points * 0.2

                # Ball control
                if 'offensive_assistTurnoverRatio' in team:
                    ast_to_ratio = float(team['offensive_assistTurnoverRatio'])
                    strength += min(ast_to_ratio / 2.0, 1.0) * 0.1  # Cap at 2.0 ratio

                # Rebounding
                if 'general_avgRebounds' in team:
                    normalized_rebounds = min(float(team['general_avgRebounds']) / 45.0, 1.0)
                    strength += normalized_rebounds * 0.1

                # Defense
                if 'defensive_avgSteals' in team:
                    normalized_steals = min(float(team['defensive_avgSteals']) / 10.0, 1.0)
                    strength += normalized_steals * 0.05

                if 'defensive_avgBlocks' in team:
                    normalized_blocks = min(float(team['defensive_avgBlocks']) / 6.0, 1.0)
                    strength += normalized_blocks * 0.05

                # Team record (if available)
                if 'record' in team:
                    try:
                        wins, losses = team['record'].split('-')
                        win_pct = float(wins) / (float(wins) + float(losses))
                        strength += win_pct * 0.2
                    except:
                        # Default if record parsing fails
                        strength += 0.5 * 0.2

                # Add conference strength bonus if we can match it
                conference_bonus = self.CONFERENCE_STRENGTH.get('default', 0)
                if self.teams_info is not None:
                    team_info = self.teams_info[self.teams_info['id'] == team_id]
                    if not team_info.empty:
                        conf = team_info['conference'].iloc[0]
                        conference_bonus = self.CONFERENCE_STRENGTH.get(conf, self.CONFERENCE_STRENGTH['default'])

                strength += conference_bonus

            except Exception as e:
                print(f"Error calculating strength for team {team_id}: {e}")
                strength = 0.5  # Default if calculation fails

            # Add some randomness to account for unmeasured factors
            strength += random.uniform(-0.05, 0.05)

            # Store the strength value (capped between 0 and 1)
            self.team_strength[team_id] = max(0.1, min(strength, 1.0))

        # If no team statistics available, create mock strengths based on rankings
        if not self.team_strength and hasattr(self, 'rankings'):
            self._create_mock_strengths()

    def _create_mock_strengths(self):
        """Create estimated team strengths based on rankings"""
        if not hasattr(self, 'rankings') or self.rankings is None:
            # Create completely mock strengths
            for i in range(1, 69):
                # Teams 1-10: 0.8-0.95, Teams 11-30: 0.65-0.8, Teams 31-50: 0.5-0.65, Teams 51-68: 0.35-0.5
                base = 0.95 - (i / 68) * 0.6
                self.team_strength[str(i)] = max(0.1, min(base + random.uniform(-0.05, 0.05), 0.95))
            return

        for _, team in self.rankings.iterrows():
            team_id = str(team['team_id'])
            rank = int(team['rank'])

            # Higher ranked teams get higher base strength
            # Scale from 0.6 (rank 25) to 0.9 (rank 1)
            if rank <= 25:
                base_strength = 0.9 - ((rank - 1) * 0.012)
            else:
                # Teams outside top 25 get lower strength
                base_strength = 0.6 - ((rank - 25) * 0.01)
                base_strength = max(base_strength, 0.2)  # Floor at 0.2

            # Add randomness
            strength = base_strength + random.uniform(-0.05, 0.05)
            self.team_strength[team_id] = max(0.1, min(strength, 1.0))

    def get_tournament_teams(self):
        """
        Get the teams in the tournament with their seeds and regions.
        Returns a dictionary organized by region and seed.
        """
        tournament = defaultdict(dict)

        # Try to use bracketology data if available
        if hasattr(self, 'bracket_data') and self.bracket_data is not None and not getattr(self.bracket_data, 'empty', True):
            for _, team in self.bracket_data.iterrows():
                region = team['region']
                seed = int(team['seed'])
                team_id = str(team['team_id'])
                team_name = team['team_name']

                tournament[region][seed] = {
                    'team_id': team_id,
                    'team_name': team_name,
                    'seed': seed
                }

            return tournament

        # If no bracketology data, create a mock bracket from rankings
        if hasattr(self, 'rankings') and self.rankings is not None and not getattr(self.rankings, 'empty', True):
            # Get top 68 teams from rankings
            top_teams = self.rankings
            if len(top_teams) > 68:
                top_teams = top_teams.head(68)

            # Define 4 regions
            regions = ['East', 'West', 'South', 'Midwest']

            # Define seeds (1-16 for each region)
            seeds = list(range(1, 17))

            # Assign teams to seeds following serpentine pattern
            for i, (_, team) in enumerate(top_teams.iterrows()):
                region_index = i % 4
                seed_index = (i // 4) + 1

                if seed_index > 16:  # Handle play-in games as 16 seeds
                    seed_index = 16

                region = regions[region_index]
                team_id = str(team['team_id'])
                team_name = team.get('team_name', '') or self._get_team_name(team_id)

                tournament[region][seed_index] = {
                    'team_id': team_id,
                    'team_name': team_name,
                    'seed': seed_index
                }

            return tournament

        # Last resort - create a minimal mock bracket for demonstration
        print("Creating a mock tournament bracket for demonstration.")

        regions = ['East', 'West', 'South', 'Midwest']
        for region in regions:
            for seed in range(1, 17):
                team_id = f"{seed + ((regions.index(region)) * 16)}"
                team_name = f"Team {team_id}"

                tournament[region][seed] = {
                    'team_id': team_id,
                    'team_name': team_name,
                    'seed': seed
                }

                # Ensure this team has a strength value
                if team_id not in self.team_strength:
                    self.team_strength[team_id] = max(0.1, min(1.0 - (seed * 0.05) + random.uniform(-0.05, 0.05), 0.95))

        return tournament

    def simulate_game(self, team1, team2, round_num=1):
        """
        Simulate a single game between two teams.

        Args:
            team1: Dictionary with team1 info including team_id and seed
            team2: Dictionary with team2 info including team_id and seed
            round_num: Tournament round number (1-6, where 6 is championship)

        Returns:
            The winning team dictionary
        """
        team1_id = str(team1['team_id'])
        team2_id = str(team2['team_id'])
        team1_seed = team1['seed']
        team2_seed = team2['seed']

        # Get base team strengths
        team1_strength = self.team_strength.get(team1_id, 0.5)
        team2_strength = self.team_strength.get(team2_id, 0.5)

        # Adjust for historical upset rates
        seed_diff = abs(team1_seed - team2_seed)
        if seed_diff > 0:
            lower_seed = min(team1_seed, team2_seed)
            higher_seed = max(team1_seed, team2_seed)

            # Get historical upset rate if available
            upset_rate = self.HISTORICAL_UPSET_RATES.get((lower_seed, higher_seed), 0)

            # Apply upset adjustment
            if team1_seed > team2_seed:  # team1 is the underdog
                team1_strength += self.UPSET_FACTOR * upset_rate
            else:  # team2 is the underdog
                team2_strength += self.UPSET_FACTOR * upset_rate

        # Adjust for tournament pressure (higher seeds handle it better in later rounds)
        if round_num >= 4:  # Sweet 16 and beyond
            pressure_factor = 0.02 * (round_num - 3)  # Increases with each round
            if team1_seed < team2_seed:
                team1_strength += pressure_factor
            else:
                team2_strength += pressure_factor

        # Randomize for simulation variance
        # Later rounds have less randomness (better teams prevail)
        randomness = max(0.3 - (round_num * 0.04), 0.1)
        team1_score = team1_strength + random.uniform(-randomness, randomness)
        team2_score = team2_strength + random.uniform(-randomness, randomness)

        # Determine winner
        if team1_score > team2_score:
            return team1
        else:
            return team2

    def simulate_tournament(self, num_simulations=1):
      """
      Simulate the entire March Madness tournament with proper NCAA bracket structure.

      Args:
          num_simulations: Number of simulations to run (default: 1)

      Returns:
          A list of dictionaries containing the results of each simulation
      """
      all_results = []

      for sim_num in range(num_simulations):
          if num_simulations > 10 and sim_num % 10 == 0:
              print(f"Running simulation {sim_num+1}/{num_simulations}")

          # Get fresh bracket
          bracket = self.get_tournament_teams()

          # Check if we have a valid bracket
          if not bracket:
              print("Error: Could not create bracket. Please check data files.")
              return []

          # Structure to hold results
          results = {
              'simulation': sim_num + 1,
              'rounds': {
                  'round_1': [],  # First Round
                  'round_2': [],  # Second Round
                  'round_3': [],  # Sweet 16
                  'round_4': [],  # Elite 8
                  'round_5': [],  # Final Four
                  'round_6': []   # Championship
              },
              'champion': None
          }

          # Dictionary to track winners by region and seed position (for bracket tracking)
          regional_winners = {region: {} for region in bracket.keys()}

          # First round matchups - these define the proper NCAA bracket structure
          first_round_matchups = [
              (1, 16), (8, 9),    # These winners play each other in round 2
              (5, 12), (4, 13),   # These winners play each other in round 2
              (6, 11), (3, 14),   # These winners play each other in round 2
              (7, 10), (2, 15)    # These winners play each other in round 2
          ]

          # Second round matchups (based on first round)
          second_round_matchups = [
              ((1, 16), (8, 9)),     # Upper bracket - left side
              ((5, 12), (4, 13)),    # Upper bracket - right side
              ((6, 11), (3, 14)),    # Lower bracket - left side
              ((7, 10), (2, 15))     # Lower bracket - right side
          ]

          # Sweet 16 matchups (based on second round)
          sweet16_matchups = [
              (((1, 16), (8, 9)), ((5, 12), (4, 13))),       # Upper bracket
              (((6, 11), (3, 14)), ((7, 10), (2, 15)))       # Lower bracket
          ]

          # FIRST ROUND - Play all first round games
          for region, seeds in bracket.items():
              for seed1, seed2 in first_round_matchups:
                  # Skip if either team is missing
                  if seed1 not in seeds or seed2 not in seeds:
                      continue

                  team1 = seeds[seed1]
                  team2 = seeds[seed2]

                  # Simulate the game
                  winner = self.simulate_game(team1, team2, round_num=1)

                  # Record the result
                  game_result = {
                      'region': region,
                      'team1': {'team_id': team1['team_id'], 'team_name': team1['team_name'], 'seed': team1['seed']},
                      'team2': {'team_id': team2['team_id'], 'team_name': team2['team_name'], 'seed': team2['seed']},
                      'winner': {'team_id': winner['team_id'], 'team_name': winner['team_name'], 'seed': winner['seed']}
                  }

                  results['rounds']['round_1'].append(game_result)

                  # Track first round winners by seed position for proper bracket advancement
                  regional_winners[region][winner['seed']] = winner

          # SECOND ROUND - Match winners according to bracket structure
          for region in bracket.keys():
              for ((seed1_a, seed1_b), (seed2_a, seed2_b)) in second_round_matchups:
                  # Find winner of the first matchup
                  team1 = None
                  if seed1_a in regional_winners[region]:
                      team1 = regional_winners[region][seed1_a]
                  elif seed1_b in regional_winners[region]:
                      team1 = regional_winners[region][seed1_b]

                  # Find winner of the second matchup
                  team2 = None
                  if seed2_a in regional_winners[region]:
                      team2 = regional_winners[region][seed2_a]
                  elif seed2_b in regional_winners[region]:
                      team2 = regional_winners[region][seed2_b]

                  # Skip if either team is missing
                  if not team1 or not team2:
                      continue

                  # Simulate the game
                  winner = self.simulate_game(team1, team2, round_num=2)

                  # Record the result
                  game_result = {
                      'region': region,
                      'team1': {'team_id': team1['team_id'], 'team_name': team1['team_name'], 'seed': team1['seed']},
                      'team2': {'team_id': team2['team_id'], 'team_name': team2['team_name'], 'seed': team2['seed']},
                      'winner': {'team_id': winner['team_id'], 'team_name': winner['team_name'], 'seed': winner['seed']}
                  }

                  results['rounds']['round_2'].append(game_result)

                  # Track second round winners for Sweet 16
                  regional_winners[region][f"R2_{winner['seed']}"] = winner

          # SWEET 16 - Match winners according to bracket structure
          for region in bracket.keys():
              for i, (matchup1, matchup2) in enumerate(sweet16_matchups):
                  # Identify the first matchup's potential seeds
                  seeds1 = []
                  for seed_pair in matchup1:
                      seeds1.extend(seed_pair)

                  # Identify the second matchup's potential seeds
                  seeds2 = []
                  for seed_pair in matchup2:
                      seeds2.extend(seed_pair)

                  # Find the team that advanced from first group
                  team1 = None
                  for seed in seeds1:
                      if seed in regional_winners[region]:
                          team1 = regional_winners[region][seed]
                          break
                      # Check in R2 keys
                      r2_key = f"R2_{seed}"
                      if r2_key in regional_winners[region]:
                          team1 = regional_winners[region][r2_key]
                          break

                  # Find the team that advanced from second group
                  team2 = None
                  for seed in seeds2:
                      if seed in regional_winners[region]:
                          team2 = regional_winners[region][seed]
                          break
                      # Check in R2 keys
                      r2_key = f"R2_{seed}"
                      if r2_key in regional_winners[region]:
                          team2 = regional_winners[region][r2_key]
                          break

                  # Skip if either team is missing
                  if not team1 or not team2:
                      continue

                  # Simulate the game
                  winner = self.simulate_game(team1, team2, round_num=3)

                  # Record the result
                  game_result = {
                      'region': region,
                      'team1': {'team_id': team1['team_id'], 'team_name': team1['team_name'], 'seed': team1['seed']},
                      'team2': {'team_id': team2['team_id'], 'team_name': team2['team_name'], 'seed': team2['seed']},
                      'winner': {'team_id': winner['team_id'], 'team_name': winner['team_name'], 'seed': winner['seed']}
                  }

                  results['rounds']['round_3'].append(game_result)

                  # Track Sweet 16 winners for Elite 8
                  regional_winners[region][f"R3_{i}"] = winner

          # ELITE 8 - Regional finals (one game per region)
          for region in bracket.keys():
              if "R3_0" in regional_winners[region] and "R3_1" in regional_winners[region]:
                  team1 = regional_winners[region]["R3_0"]
                  team2 = regional_winners[region]["R3_1"]

                  # Simulate the game
                  winner = self.simulate_game(team1, team2, round_num=4)

                  # Record the result
                  game_result = {
                      'region': region,
                      'team1': {'team_id': team1['team_id'], 'team_name': team1['team_name'], 'seed': team1['seed']},
                      'team2': {'team_id': team2['team_id'], 'team_name': team2['team_name'], 'seed': team2['seed']},
                      'winner': {'team_id': winner['team_id'], 'team_name': winner['team_name'], 'seed': winner['seed']}
                  }

                  results['rounds']['round_4'].append(game_result)

                  # Store regional champion
                  winner['region'] = region
                  regional_winners[region]["champion"] = winner

          # FINAL FOUR - Cross-regional matchups (standard pairings)
          # Standard NCAA bracket has: regions[0] vs regions[1], regions[2] vs regions[3]
          regions = list(bracket.keys())
          if len(regions) >= 4:
              final_four_teams = []
              for region in regions:
                  if "champion" in regional_winners[region]:
                      final_four_teams.append(regional_winners[region]["champion"])

              if len(final_four_teams) >= 4:
                  # First semifinal
                  team1 = final_four_teams[0]
                  team2 = final_four_teams[1]

                  winner = self.simulate_game(team1, team2, round_num=5)
                  winner['region'] = team1['region'] if winner['team_id'] == team1['team_id'] else team2['region']

                  game_result = {
                      'team1': {'team_id': team1['team_id'], 'team_name': team1['team_name'],
                              'seed': team1['seed'], 'region': team1['region']},
                      'team2': {'team_id': team2['team_id'], 'team_name': team2['team_name'],
                              'seed': team2['seed'], 'region': team2['region']},
                      'winner': {'team_id': winner['team_id'], 'team_name': winner['team_name'],
                                'seed': winner['seed'], 'region': winner['region']}
                  }

                  results['rounds']['round_5'].append(game_result)
                  finalist1 = winner

                  # Second semifinal
                  team1 = final_four_teams[2]
                  team2 = final_four_teams[3]

                  winner = self.simulate_game(team1, team2, round_num=5)
                  winner['region'] = team1['region'] if winner['team_id'] == team1['team_id'] else team2['region']

                  game_result = {
                      'team1': {'team_id': team1['team_id'], 'team_name': team1['team_name'],
                              'seed': team1['seed'], 'region': team1['region']},
                      'team2': {'team_id': team2['team_id'], 'team_name': team2['team_name'],
                              'seed': team2['seed'], 'region': team2['region']},
                      'winner': {'team_id': winner['team_id'], 'team_name': winner['team_name'],
                                'seed': winner['seed'], 'region': winner['region']}
                  }

                  results['rounds']['round_5'].append(game_result)
                  finalist2 = winner

                  # Championship game
                  champion = self.simulate_game(finalist1, finalist2, round_num=6)
                  champion['region'] = finalist1['region'] if champion['team_id'] == finalist1['team_id'] else finalist2['region']

                  game_result = {
                      'team1': {'team_id': finalist1['team_id'], 'team_name': finalist1['team_name'],
                              'seed': finalist1['seed'], 'region': finalist1['region']},
                      'team2': {'team_id': finalist2['team_id'], 'team_name': finalist2['team_name'],
                              'seed': finalist2['seed'], 'region': finalist2['region']},
                      'winner': {'team_id': champion['team_id'], 'team_name': champion['team_name'],
                                'seed': champion['seed'], 'region': champion['region']}
                  }

                  results['rounds']['round_6'].append(game_result)
                  results['champion'] = champion

          all_results.append(results)

      return all_results

    def run_simulation(self, num_simulations=100):
        """
        Run multiple tournament simulations and analyze the results.

        Args:
            num_simulations: Number of simulations to run

        Returns:
            Dictionary with aggregated simulation results
        """
        print(f"Running {num_simulations} March Madness simulations...")

        sim_results = self.simulate_tournament(num_simulations)

        if not sim_results:
            return {"error": "Simulation failed to run."}

        # Aggregate results
        champion_counts = defaultdict(int)
        final_four_counts = defaultdict(int)
        elite_eight_counts = defaultdict(int)
        upset_counts = []  # Track upsets by round

        for result in sim_results:
            # Count champions
            if 'champion' in result and result['champion']:
                champion_id = result['champion']['team_id']
                champion_counts[champion_id] += 1

            # Count Final Four appearances
            for game in result['rounds'].get('round_5', []):
                if 'team1' in game and game['team1'] is not None:
                    final_four_counts[game['team1']['team_id']] += 1
                if 'team2' in game and game['team2'] is not None:
                    final_four_counts[game['team2']['team_id']] += 1

            # Count Elite Eight appearances
            for game in result['rounds'].get('round_4', []):
                if 'team1' in game and game['team1'] is not None:
                    elite_eight_counts[game['team1']['team_id']] += 1
                if 'team2' in game and game['team2'] is not None:
                    elite_eight_counts[game['team2']['team_id']] += 1

            # Count upsets by round
            for round_key, games in result['rounds'].items():
                for game in games:
                    # Skip games with missing data
                    if ('winner' not in game) or ('team1' not in game) or ('team2' not in game):
                        continue

                    # An upset is when the higher seed beats the lower seed
                    if game['winner']['seed'] > min(game['team1']['seed'], game['team2']['seed']):
                        upset_counts.append({
                            'round': round_key,
                            'winner_seed': game['winner']['seed'],
                            'loser_seed': game['team1']['seed'] if game['winner']['team_id'] == game['team2']['team_id'] else game['team2']['seed']
                        })

        # Convert champion counts to percentages
        champion_percentages = {}
        for team_id, count in champion_counts.items():
            team_name = self._get_team_name(team_id)
            champion_percentages[team_name] = (count / num_simulations) * 100

        # Sort by highest percentage
        champion_percentages = dict(sorted(champion_percentages.items(), key=lambda item: item[1], reverse=True))

        # Similar for Final Four and Elite Eight
        final_four_percentages = {}
        for team_id, count in final_four_counts.items():
            team_name = self._get_team_name(team_id)
            final_four_percentages[team_name] = (count / (num_simulations * 2)) * 100  # Divide by 2 because each team appears twice

        elite_eight_percentages = {}
        for team_id, count in elite_eight_counts.items():
            team_name = self._get_team_name(team_id)
            elite_eight_percentages[team_name] = (count / (num_simulations * 2)) * 100  # Divide by 2 because each team appears twice

        # Sort by highest percentage
        final_four_percentages = dict(sorted(final_four_percentages.items(), key=lambda item: item[1], reverse=True))
        elite_eight_percentages = dict(sorted(elite_eight_percentages.items(), key=lambda item: item[1], reverse=True))

        # Count upsets by round
        upset_summary = defaultdict(int)
        for upset in upset_counts:
            upset_summary[upset['round']] += 1

        avg_upsets_per_sim = len(upset_counts) / num_simulations

        # Detailed results from last simulation (for bracket visualization)
        last_sim = sim_results[-1]

        # Compile all results
        aggregated_results = {
            'num_simulations': num_simulations,
            'champion_percentages': champion_percentages,
            'final_four_percentages': final_four_percentages,
            'elite_eight_percentages': elite_eight_percentages,
            'upsets': {
                'avg_upsets_per_sim': avg_upsets_per_sim,
                'upsets_by_round': dict(upset_summary)
            },
            'last_simulation': last_sim
        }

        return aggregated_results

    def _get_team_name(self, team_id):
        """Helper to get team name from ID"""
        # Use our custom mapping first
        team_mapping = {
            "1": "Auburn",
            "2": "Houston",
            "3": "Florida",
            "4": "Duke",
            "5": "Michigan St.",
            "6": "Wofford",
            "7": "Saint John's",
            "8": "Alabama",
            "9": "Iowa St.",
            "10": "Marquette",
            "11": "Texas Tech",
            "12": "Wisconsin",
            "13": "Texas A&M",
            "14": "Purdue",
            "15": "Maryland",
            "16": "Arizona",
            "17": "Kansas",
            "18": "Clemson",
            "19": "Memphis",
            "20": "Oregon",
            "23": "Missouri",
            "33": "Creighton",
            "34": "Georgetown",
            "35": "Oklahoma",
            "36": "Baylor",
            "37": "New Mexico",
            "38": "Tennessee",
            "39": "Arkansas",
            "40": "Vanderbilt",
            "41": "North Carolina",
            "42": "Xavier",
            "44": "VCU",
            "45": "UC San Diego",
            "54": "Troy"
        }

        if str(team_id) in team_mapping:
            return team_mapping[str(team_id)]

        # Then try the usual lookup methods
        team_id = str(team_id)

        # Try to find in teams info
        if hasattr(self, 'teams_info') and self.teams_info is not None:
            team = self.teams_info[self.teams_info['id'].astype(str) == team_id]
            if not team.empty:
                return team['name'].iloc[0]

        # Try to find in rankings
        if hasattr(self, 'rankings') and self.rankings is not None:
            team = self.rankings[self.rankings['team_id'].astype(str) == team_id]
            if not team.empty:
                return team['team_name'].iloc[0]

        # Try to find in team stats
        if hasattr(self, 'team_stats') and self.team_stats is not None:
            team = self.team_stats[self.team_stats['team_id'].astype(str) == team_id]
            if not team.empty:
                return team['team_name'].iloc[0]

        return f"Team {team_id}"  # Default if not found

    def print_summary(self, results):
        """Print a readable summary of simulation results"""
        print("\n===== MARCH MADNESS SIMULATION SUMMARY =====")
        print(f"Number of simulations: {results['num_simulations']}")

        print("\n=== CHAMPIONSHIP ODDS ===")
        print("Team, Championship %")
        for team, pct in list(results['champion_percentages'].items())[:10]:  # Top 10
            print(f"{team}: {pct:.1f}%")

        print("\n=== FINAL FOUR ODDS ===")
        print("Team, Final Four %")
        for team, pct in list(results['final_four_percentages'].items())[:10]:  # Top 10
            print(f"{team}: {pct:.1f}%")

        print("\n=== UPSET SUMMARY ===")
        print(f"Average upsets per tournament: {results['upsets']['avg_upsets_per_sim']:.1f}")
        for round_name, count in results['upsets']['upsets_by_round'].items():
            print(f"  {round_name}: {count / results['num_simulations']:.1f} upsets per tournament")

        print("\n=== LAST SIMULATION BRACKET ===")

        # Add defensive checks for null values
        if 'last_simulation' in results and results['last_simulation'] is not None:
            last_sim = results['last_simulation']

            # Check if champion exists
            if 'champion' in last_sim and last_sim['champion'] is not None:
                print(f"Champion: {last_sim['champion']['seed']} {last_sim['champion']['team_name']} ({last_sim['champion']['region']})")
            else:
                print("Champion: None determined")

            print("\nFinal Four Teams:")
            if 'rounds' in last_sim and 'round_5' in last_sim['rounds']:
                for game in last_sim['rounds']['round_5']:
                    print(f"  {game['team1']['seed']} {game['team1']['team_name']} vs {game['team2']['seed']} {game['team2']['team_name']}")
                    print(f"  Winner: {game['winner']['seed']} {game['winner']['team_name']}")
            else:
                print("  No Final Four teams determined")
        else:
            print("No simulation results available")

        print("\n===========================================")

    def save_results(self, results, filename="march_madness_simulation.json"):
        """Save simulation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")

    def generate_bracket_visualization(self, results):
        """Generate a text-based bracket visualization from simulation results"""
        # Handle different input formats
        if 'last_simulation' in results:
            # Format from generate_consensus_bracket
            last_sim = results['last_simulation']
        else:
            # Original format from run_simulation
            last_sim = results['last_simulation'] if 'last_simulation' in results else results[-1]
    
        bracket = "========== MARCH MADNESS SIMULATED BRACKET ==========\n\n"

        # Regions
        regions = set()
        for game in last_sim['rounds']['round_1']:
            regions.add(game['region'])

        for region in regions:
            bracket += f"===== {region} REGION =====\n"

            # Find winners by round in this region
            winners_by_round = {}
            for round_num in range(1, 5):  # Rounds 1-4
                round_key = f'round_{round_num}'
                winners = []

                for game in last_sim['rounds'][round_key]:
                    if 'region' in game and game['region'] == region:
                        winners.append(game['winner'])

                winners_by_round[round_num] = winners

            # Print rounds
            round_names = {
                1: "First Round",
                2: "Second Round",
                3: "Sweet 16",
                4: "Elite Eight"
            }

            for round_num, name in round_names.items():
                if round_num in winners_by_round:
                    bracket += f"\n* {name} *\n"
                    for winner in winners_by_round[round_num]:
                        bracket += f"({winner['seed']}) {winner['team_name']}\n"

            bracket += "\n"

        # Final Four
        bracket += "===== FINAL FOUR =====\n\n"
        for game in last_sim['rounds']['round_5']:
            bracket += f"({game['team1']['seed']}) {game['team1']['team_name']} vs "
            bracket += f"({game['team2']['seed']}) {game['team2']['team_name']}\n"
            bracket += f"Winner: ({game['winner']['seed']}) {game['winner']['team_name']}\n\n"

        # Championship
        bracket += "===== CHAMPIONSHIP =====\n\n"
        if last_sim['rounds']['round_6']:
            game = last_sim['rounds']['round_6'][0]
            bracket += f"({game['team1']['seed']}) {game['team1']['team_name']} vs "
            bracket += f"({game['team2']['seed']}) {game['team2']['team_name']}\n"
            bracket += f"Winner: ({game['winner']['seed']}) {game['winner']['team_name']}\n\n"

        bracket += "===== CHAMPION =====\n"
        if last_sim['champion']:
            bracket += f"({last_sim['champion']['seed']}) {last_sim['champion']['team_name']}\n"

        return bracket

    def create_championship_odds_chart(self, results, filename="championship_odds.png"):
        """Create a bar chart of championship odds"""
        # Get top teams
        top_teams = dict(list(results['champion_percentages'].items())[:15])

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create horizontal bar chart
        teams = list(top_teams.keys())
        percentages = list(top_teams.values())

        # Create color gradient based on odds (higher = darker blue)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(teams)))

        # Create horizontal bar chart with teams sorted by percentages
        bars = plt.barh(teams, percentages, color=colors)

        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                     f'{width:.1f}%', ha='left', va='center')

        # Add title and labels
        plt.title('NCAA Tournament Championship Odds', fontsize=16)
        plt.xlabel('Probability (%)', fontsize=12)
        plt.ylabel('Team', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved championship odds chart to {filename}")
        plt.close()

    def create_final_four_odds_chart(self, results, filename="final_four_odds.png"):
        """Create a bar chart of Final Four odds"""
        # Get top teams
        top_teams = dict(list(results['final_four_percentages'].items())[:20])

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create horizontal bar chart
        teams = list(top_teams.keys())
        percentages = list(top_teams.values())

        # Create color gradient based on odds (higher = darker green)
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(teams)))

        # Create horizontal bar chart with teams sorted by percentages
        bars = plt.barh(teams, percentages, color=colors)

        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                     f'{width:.1f}%', ha='left', va='center')

        # Add title and labels
        plt.title('NCAA Tournament Final Four Odds', fontsize=16)
        plt.xlabel('Probability (%)', fontsize=12)
        plt.ylabel('Team', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved Final Four odds chart to {filename}")
        plt.close()

    def visualize_results(self, results):
        """Create all visualizations for simulation results"""
        print("Creating visualizations...")
        self.create_championship_odds_chart(results)
        self.create_final_four_odds_chart(results)

        # Generate text bracket
        bracket = self.generate_bracket_visualization(results)
        with open("simulated_bracket.txt", "w") as f:
            f.write(bracket)
        print("Saved bracket visualization to simulated_bracket.txt")


    def generate_consensus_bracket(self, simulation_results):
        """Alias method for backtesting compatibility"""
        return {'last_simulation': simulation_results[-1]}
