# Advanced NCAA Tournament Prediction Metrics
# Based on KenPom, BPI, and other renowned systems

import pandas as pd
import numpy as np
from collections import defaultdict
import math

class AdvancedMetricsCalculator:
    """Calculate advanced predictive metrics for NCAA tournament teams"""

    def __init__(self, team_stats_df, national_averages=None):
        """
        Initialize with team statistics dataframe

        Args:
            team_stats_df: DataFrame containing team statistics
            national_averages: Optional dict with national averages for normalization
        """
        self.team_stats = team_stats_df

        # Set default national averages if not provided (based on typical NCAA averages)
        if national_averages is None:
            self.national_averages = {
                'offensive_avgPoints': 73.0,
                'offensive_fieldGoalPct': 44.0,
                'offensive_threePointFieldGoalPct': 34.0,
                'offensive_freeThrowPct': 71.0,
                'defensive_avgSteals': 6.5,
                'defensive_avgBlocks': 3.5,
                'general_avgRebounds': 36.0,
                'offensive_avgAssists': 13.0,
                'offensive_avgTurnovers': 12.5,
                'possessions_per_game': 68.0  # Typical NCAA average
            }
        else:
            self.national_averages = national_averages

        # Calculate national averages from the provided data if we have enough teams
        if len(team_stats_df) > 100:  # Only if we have a substantial sample
            print("Calculating national averages from data...")
            self._calculate_national_averages()

        # Store conference strength data
        self.conference_strength = {}

        # Store the calculated metrics
        self.team_metrics = {}

    def _calculate_national_averages(self):
        """Calculate national averages for key statistics from our data"""
        for stat in self.national_averages.keys():
            if stat in self.team_stats.columns:
                try:
                    avg_value = self.team_stats[stat].astype(float).mean()
                    if not np.isnan(avg_value):
                        self.national_averages[stat] = avg_value
                except:
                    # Skip if conversion issues
                    pass

        # Print the national averages we're using
        print("Using national averages:")
        for stat, value in self.national_averages.items():
            print(f"  {stat}: {value:.2f}")

    def calculate_all_metrics(self):
        """Calculate all advanced metrics for all teams"""
        print("Calculating advanced metrics for all teams...")

        # First calculate conference strength
        self.calculate_conference_strength()

        # Calculate metrics for each team
        for _, team in self.team_stats.iterrows():
            team_id = str(team['team_id'])
            try:
                metrics = self.calculate_team_metrics(team)
                self.team_metrics[team_id] = metrics
            except Exception as e:
                print(f"Error calculating metrics for team {team_id}: {e}")
                # Provide default metrics for teams with errors
                self.team_metrics[team_id] = {
                    'adjusted_efficiency': 0.5,
                    'tempo_adjusted_rating': 0.5,
                    'defensive_efficiency': 0.5,
                    'experience_factor': 0.5,
                    'consistency_rating': 0.5,
                    'tournament_readiness': 0.5,
                    'overall_power_index': 0.5
                }

        return self.team_metrics

    def calculate_conference_strength(self):
        """Calculate conference strength based on non-conference performance"""
        # Group teams by conference
        conferences = defaultdict(list)

        # Try to extract conference from data
        for _, team in self.team_stats.iterrows():
            if 'conference' in team:
                conf = team['conference']
                conferences[conf].append(team)

        # Calculate average strength
        for conf, teams in conferences.items():
            win_pct = 0
            offensive_rating = 0
            defensive_rating = 0

            # Calculate average performance
            for team in teams:
                # Win percentage
                if 'record' in team:
                    try:
                        wins, losses = team['record'].split('-')
                        win_pct += float(wins) / (float(wins) + float(losses))
                    except:
                        win_pct += 0.5  # Default if parse fails

                # Offensive and defensive ratings approximation
                if 'offensive_avgPoints' in team:
                    offensive_rating += float(team['offensive_avgPoints'])

                if 'defensive_avgSteals' in team and 'defensive_avgBlocks' in team:
                    defensive_rating += float(team['defensive_avgSteals']) + float(team['defensive_avgBlocks'])

            # Calculate averages
            n_teams = len(teams)
            if n_teams > 0:
                win_pct /= n_teams
                offensive_rating /= n_teams
                defensive_rating /= n_teams

                # Combine into overall strength
                strength = (
                    (win_pct - 0.5) * 2 +  # Normalize to [-1, 1]
                    (offensive_rating / 80.0 - 0.5) +  # Normalize assuming 80 is excellent
                    (defensive_rating / 10.0 - 0.5)  # Normalize assuming 10 is excellent
                ) / 3  # Average the components

                # Store normalized to [0, 1]
                self.conference_strength[conf] = max(0.0, min(1.0, strength + 0.5))
            else:
                self.conference_strength[conf] = 0.5  # Default

    def calculate_team_metrics(self, team):
        """
        Calculate advanced metrics for a single team

        Args:
            team: Series representing a single team's stats

        Returns:
            Dictionary of calculated metrics
        """
        team_id = str(team['team_id'])
        metrics = {}

        # 1. Calculate Offensive Efficiency (points per 100 possessions)
        # Estimate possessions
        possessions = self._estimate_possessions(team)

        # Calculate offensive efficiency (points per 100 possessions)
        if 'offensive_avgPoints' in team and possessions > 0:
            metrics['offensive_efficiency'] = float(team['offensive_avgPoints']) * 100 / possessions
        else:
            metrics['offensive_efficiency'] = 50.0  # Default

        # 2. Calculate Defensive Efficiency (points allowed per 100 possessions)
        # We may not have opponent points, so we'll estimate based on available defensive stats
        defensive_rating = 0
        if 'defensive_avgSteals' in team:
            defensive_rating += float(team['defensive_avgSteals']) * 5  # Each steal worth about 5 points

        if 'defensive_avgBlocks' in team:
            defensive_rating += float(team['defensive_avgBlocks']) * 3  # Each block worth about 3 points

        # Convert to a defensive efficiency (lower is better)
        metrics['defensive_efficiency'] = max(80, 110 - defensive_rating)

        # 3. Adjusted Efficiency Margin (similar to KenPom)
        oe_adj = metrics['offensive_efficiency'] / 100.0  # Normalize to 0-1
        de_adj = (120 - metrics['defensive_efficiency']) / 40.0  # Normalize to 0-1 (lower is better)
        metrics['adjusted_efficiency'] = (oe_adj + de_adj) / 2

        # 4. Tempo-Adjusted Rating
        if possessions > 0:
            tempo_factor = possessions / self.national_averages['possessions_per_game']
            metrics['tempo_adjusted_rating'] = metrics['adjusted_efficiency'] * (1 + (tempo_factor - 1) * 0.2)
        else:
            metrics['tempo_adjusted_rating'] = metrics['adjusted_efficiency']

        # 5. Experience Factor (using record as a proxy)
        win_pct = 0.5  # Default
        if 'record' in team:
            try:
                wins, losses = team['record'].split('-')
                win_pct = float(wins) / (float(wins) + float(losses))
            except:
                pass

        # More experienced teams likely have better records
        metrics['experience_factor'] = win_pct * 0.8 + 0.2  # Ensure factor is at least 0.2

        # 6. Consistency Rating (using assist/turnover ratio as proxy)
        consistency = 0.5  # Default
        if 'offensive_assistTurnoverRatio' in team:
            ratio = float(team['offensive_assistTurnoverRatio'])
            # Higher ratio = more consistent
            consistency = min(1.0, ratio / 2.0)  # Cap at 1.0

        metrics['consistency_rating'] = consistency

        # 7. Tournament Readiness Score (combines key tournament success factors)
        tournament_factors = [
            metrics['adjusted_efficiency'] * 0.4,  # Efficiency is critical
            metrics['experience_factor'] * 0.2,    # Experience matters
            metrics['consistency_rating'] * 0.2,   # Consistency is important
        ]

        # Add conference strength if available
        if 'conference' in team and team['conference'] in self.conference_strength:
            conf_strength = self.conference_strength[team['conference']]
            tournament_factors.append(conf_strength * 0.2)
        else:
            tournament_factors.append(0.1)  # Default conference boost

        metrics['tournament_readiness'] = sum(tournament_factors)

        # 8. Overall Power Index (combined rating)
        metrics['overall_power_index'] = (
            metrics['adjusted_efficiency'] * 0.4 +
            metrics['tempo_adjusted_rating'] * 0.2 +
            metrics['experience_factor'] * 0.2 +
            metrics['consistency_rating'] * 0.1 +
            metrics['tournament_readiness'] * 0.1
        )

        # 9. Upset Potential (metric of how likely a team is to cause upsets)
        # Teams with high variance and good shooting typically cause upsets
        shooting_strength = 0.5
        if 'offensive_threePointFieldGoalPct' in team:
            three_pt_pct = float(team['offensive_threePointFieldGoalPct'])
            shooting_strength = three_pt_pct / 40.0  # Normalize (40% is excellent)

        metrics['upset_potential'] = (
            shooting_strength * 0.5 +
            metrics['adjusted_efficiency'] * 0.3 +
            (1 - metrics['consistency_rating']) * 0.2  # Less consistent teams can be upset makers
        )

        # Normalize all metrics to 0-1 range
        for key in metrics:
            metrics[key] = max(0.0, min(1.0, metrics[key]))

        return metrics

    def _estimate_possessions(self, team):
        """
        Estimate possessions per game using available stats
        Formula: FGA + 0.475*FTA - ORB + TOV

        Args:
            team: Series with team stats

        Returns:
            Estimated possessions per game
        """
        possessions = self.national_averages['possessions_per_game']  # Default

        try:
            # Components for possession estimation
            fga = float(team.get('offensive_avgFieldGoalsAttempted', 0))
            fta = float(team.get('offensive_avgFreeThrowsAttempted', 0))
            orb = float(team.get('offensive_avgOffensiveRebounds', 0))
            tov = float(team.get('offensive_avgTurnovers', 0))

            # Calculate if we have the minimum components
            if fga > 0 and tov > 0:
                possessions = fga + 0.475 * fta - orb + tov
        except:
            # Use default if calculation fails
            pass

        return possessions

    def get_team_metric(self, team_id, metric='overall_power_index'):
        """Get a specific metric for a team"""
        if team_id in self.team_metrics and metric in self.team_metrics[team_id]:
            return self.team_metrics[team_id][metric]
        else:
            return 0.5  # Default value if not found

    def get_matchup_prediction(self, team1_id, team2_id, seed1=None, seed2=None):
        """
        Calculate win probability for team1 vs team2

        Args:
            team1_id: ID of first team
            team2_id: ID of second team
            seed1: Optional seed of team1
            seed2: Optional seed of team2

        Returns:
            Probability (0-1) that team1 beats team2
        """
        # Get team metrics, use default if not found
        team1_power = self.get_team_metric(team1_id, 'overall_power_index')
        team2_power = self.get_team_metric(team2_id, 'overall_power_index')

        # Basic log5 formula for predicting matchups (used by KenPom)
        # P(A beats B) = (A - A*B) / (A + B - 2*A*B)
        # where A and B are the win probabilities against an average team

        # Convert overall power index to win probability against average team
        team1_wp = 0.5 + (team1_power - 0.5) * 2  # Scale from [0,1] to [0,1]
        team2_wp = 0.5 + (team2_power - 0.5) * 2  # Scale from [0,1] to [0,1]

        # Apply log5 formula
        if team1_wp + team2_wp != 2*team1_wp*team2_wp:  # Prevent division by zero
            win_prob = (team1_wp - team1_wp*team2_wp) / (team1_wp + team2_wp - 2*team1_wp*team2_wp)
        else:
            win_prob = 0.5  # Default to even odds if formula breaks

        # Add seed differential adjustment
        if seed1 is not None and seed2 is not None:
            # Historical seed matchup adjustment
            seed_diff = seed2 - seed1  # Positive if team1 is better seeded

            # Adjust based on seed difference (more impactful for large differences)
            seed_adj = min(0.15, max(-0.15, seed_diff * 0.015))  # Cap at ±15%
            win_prob += seed_adj

        # Get upset potential for lower seed
        if seed1 is not None and seed2 is not None:
            if seed1 > seed2:  # Team 1 is lower seed (potential upset)
                upset_potential = self.get_team_metric(team1_id, 'upset_potential')
                win_prob += upset_potential * 0.1  # Add up to 10% for high upset potential
            elif seed2 > seed1:  # Team 2 is lower seed (potential upset)
                upset_potential = self.get_team_metric(team2_id, 'upset_potential')
                win_prob -= upset_potential * 0.1  # Subtract up to 10% for high upset potential

        # Ensure probability is in valid range
        return max(0.01, min(0.99, win_prob))

# Function to enhance the simulator with advanced metrics
def enhance_simulator_with_advanced_metrics(simulator, team_stats_df=None):
    """
    Enhance the MarchMadnessSimulator with advanced metrics

    Args:
        simulator: The MarchMadnessSimulator instance
        team_stats_df: Optional DataFrame with team stats (will use simulator's if None)

    Returns:
        Updated simulator with enhanced team_strength
    """
    # Use simulator's team stats if not provided
    if team_stats_df is None and hasattr(simulator, 'team_stats'):
        team_stats_df = simulator.team_stats

    if team_stats_df is None or team_stats_df.empty:
        print("No team stats available for advanced metrics")
        return simulator

    # Calculate advanced metrics
    calculator = AdvancedMetricsCalculator(team_stats_df)
    metrics = calculator.calculate_all_metrics()

    # Update simulator's team_strength with advanced metrics
    for team_id, team_metrics in metrics.items():
        if team_id in simulator.team_strength:
            # Combine existing strength with advanced metrics
            # Weight advanced metrics more heavily (70%)
            advanced_strength = team_metrics['overall_power_index']
            original_strength = simulator.team_strength[team_id]
            simulator.team_strength[team_id] = 0.3 * original_strength + 0.7 * advanced_strength

    # Add prediction method to simulator
    simulator.calculator = calculator
    simulator.get_matchup_prediction = lambda team1_id, team2_id, seed1=None, seed2=None: calculator.get_matchup_prediction(team1_id, team2_id, seed1, seed2)

    print(f"Enhanced simulator with advanced metrics for {len(metrics)} teams")
    return simulator
