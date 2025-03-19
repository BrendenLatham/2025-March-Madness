import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
import random
from datetime import datetime

class MarchMadnessBacktester:
    """
    A framework for backtesting March Madness bracket prediction models
    against historical tournament results.
    """

    def __init__(self, base_simulator=None, years=None):
        """
        Initialize the backtester with a simulator and years to test.
        
        Args:
            base_simulator: A simulator object with at least a simulate_tournament method
            years: List of years to backtest against (None for all available)
        """
        self.base_simulator = base_simulator
        self.years = years or list(range(2010, 2023))  # Default to 2010-2022
        
        # Historical tournament data
        self.historical_data = {}
        
        # Metrics for evaluation
        self.metrics = {
            'accuracy': {},
            'bracket_score': {},
            'upset_detection': {},
            'final_four_accuracy': {},
            'champion_accuracy': {}
        }
        
        # Parameters being tested
        self.parameters = {
            'UPSET_FACTOR': 0.20,
            'MOMENTUM_FACTOR': 0.08,
            'variance_factors': {
                1: 1.0,  # First round
                2: 1.2,  # Second round
                3: 1.4,  # Sweet 16
                4: 1.6,  # Elite 8
                5: 1.8,  # Final Four
                6: 2.0   # Championship
            },
            'seed_performance_adjustments': {
                1: 1.10,   # 1-seeds
                2: 1.05,   # 2-seeds
                # ... other seeds
                12: 1.25,  # 12-seeds
                # ... rest of seeds
            },
            'historical_matchup_weights': 0.5  # 50% weight to historical data
        }
    
    def load_historical_tournaments(self, data_dir="historical_data"):
        """
        Load historical tournament data from files.
        
        Args:
            data_dir: Directory containing historical data files
        """
        print(f"Loading historical tournament data from {data_dir}...")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize with sample data if we don't have historical files
        if not os.path.exists(f"{data_dir}/tournaments.json"):
            self._initialize_sample_data(data_dir)
        
        # Load tournament results
        with open(f"{data_dir}/tournaments.json", 'r') as f:
            self.historical_data = json.load(f)
        
        print(f"Loaded data for {len(self.historical_data)} tournaments")
    
    def _initialize_sample_data(self, data_dir):
        """Create sample historical data for testing"""
        print("Creating sample historical tournament data...")
        
        sample_data = {}
        
        # 2022 Final Four teams (sample data)
        sample_data["2022"] = {
            "regions": ["West", "East", "South", "Midwest"],
            "final_four": [
                {"team_name": "Duke", "seed": 2, "region": "West"},
                {"team_name": "North Carolina", "seed": 8, "region": "East"},
                {"team_name": "Kansas", "seed": 1, "region": "Midwest"},
                {"team_name": "Villanova", "seed": 2, "region": "South"}
            ],
            "champion": {"team_name": "Kansas", "seed": 1, "region": "Midwest"},
            "rounds": {
                "round_1": [
                    # Format: [winner_seed, loser_seed, region] - partial list for sample
                    [1, 16, "West"], [8, 9, "West"], [5, 12, "West"], [4, 13, "West"],
                    [6, 11, "West"], [3, 14, "West"], [7, 10, "West"], [2, 15, "West"],
                    # East region matches...
                    [1, 16, "East"], [8, 9, "East"], [5, 12, "East"], [13, 4, "East"],
                    [11, 6, "East"], [3, 14, "East"], [7, 10, "East"], [2, 15, "East"],
                    # And so on for other regions...
                ]
                # Similar format for other rounds
            },
            "upsets": [
                # Format: [winner_seed, loser_seed, round, region]
                [13, 4, 1, "East"],
                [11, 6, 1, "East"],
                [8, 1, 3, "East"],
                # etc.
            ]
        }
        
        # Write to file
        with open(f"{data_dir}/tournaments.json", 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    def backtest_parameter(self, parameter_name, values):
        """
        Test different values for a specific parameter.
        
        Args:
            parameter_name: Name of the parameter to test
            values: List of values to test
            
        Returns:
            DataFrame with performance metrics for each value
        """
        print(f"Backtesting parameter: {parameter_name}")
        
        results = []
        original_value = None
        
        # Save original value
        if "." in parameter_name:
            # Handle nested parameters like 'variance_factors.1'
            main_param, sub_param = parameter_name.split(".")
            original_value = self.parameters[main_param][int(sub_param)]
        else:
            original_value = self.parameters[parameter_name]
        
        # Test each value
        for value in values:
            # Set parameter value
            if "." in parameter_name:
                main_param, sub_param = parameter_name.split(".")
                self.parameters[main_param][int(sub_param)] = value
            else:
                self.parameters[parameter_name] = value
            
            # Run backtests with this parameter value
            metrics = self.run_backtests()
            
            # Store results
            results.append({
                'parameter_value': value,
                'bracket_score': metrics['average_bracket_score'],
                'accuracy': metrics['average_accuracy'],
                'final_four_accuracy': metrics['final_four_accuracy'],
                'champion_accuracy': metrics['champion_accuracy']
            })
        
        # Reset to original value
        if "." in parameter_name:
            main_param, sub_param = parameter_name.split(".")
            self.parameters[main_param][int(sub_param)] = original_value
        else:
            self.parameters[parameter_name] = original_value
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print best value
        best_idx = results_df['bracket_score'].idxmax()
        best_value = results_df.loc[best_idx, 'parameter_value']
        best_score = results_df.loc[best_idx, 'bracket_score']
        print(f"Best value for {parameter_name}: {best_value} "
              f"(Score: {best_score:.2f})")
        
        return results_df
    
    def run_backtests(self, num_simulations=100):
        """
        Run backtests against multiple tournament years.
        
        Args:
            num_simulations: Number of simulations to run for each year
            
        Returns:
            Dictionary of aggregated metrics
        """
        all_metrics = defaultdict(list)
        
        for year in self.years:
            if str(year) not in self.historical_data:
                print(f"No data available for {year}, skipping...")
                continue
                
            print(f"Backtesting against {year} tournament...")
            
            # Run backtest for this year
            metrics = self.backtest_year(year, num_simulations)
            
            # Store metrics
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # Calculate averages
        average_metrics = {
            'average_bracket_score': np.mean(all_metrics['bracket_score']),
            'average_accuracy': np.mean(all_metrics['accuracy']),
            'final_four_accuracy': np.mean(all_metrics['final_four_accuracy']),
            'champion_accuracy': np.mean(all_metrics['champion_accuracy']),
            'upset_detection_rate': np.mean(all_metrics['upset_detection_rate'])
        }
        
        print(f"Overall backtesting results:")
        print(f"  Average bracket score: {average_metrics['average_bracket_score']:.2f}")
        print(f"  Average prediction accuracy: {average_metrics['average_accuracy']:.2f}%")
        print(f"  Final Four accuracy: {average_metrics['final_four_accuracy']:.2f}%")
        print(f"  Champion accuracy: {average_metrics['champion_accuracy']:.2f}%")
        print(f"  Upset detection rate: {average_metrics['upset_detection_rate']:.2f}%")
        
        return average_metrics
    
    def backtest_year(self, year, num_simulations=100):
        """
        Backtest against a specific tournament year.
        
        Args:
            year: Tournament year to test against
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary of metrics for this year
        """
        year_str = str(year)
        historical_tournament = self.historical_data[year_str]
        
        # Configure the simulator for this year
        self._configure_simulator_for_year(year)
        
        # Run simulations
        simulation_results = self._run_simulations(num_simulations)
        
        # Generate consensus bracket
        consensus_bracket = self._generate_consensus_bracket(simulation_results)
        
        # Evaluate against actual results
        metrics = self._evaluate_bracket(consensus_bracket, historical_tournament)
        
        return metrics
    
    def _configure_simulator_for_year(self, year):
        """
        Configure the simulator with appropriate data for a specific year.
        
        Args:
            year: The year to configure for
        """
        # In a real implementation, this would:
        # 1. Load team data from that year
        # 2. Set up the initial bracket structure
        # 3. Configure simulator parameters
        
        # For demo, we'll just modify the simulator parameters
        if hasattr(self.base_simulator, 'UPSET_FACTOR'):
            self.base_simulator.UPSET_FACTOR = self.parameters['UPSET_FACTOR']
        
        if hasattr(self.base_simulator, 'MOMENTUM_FACTOR'):
            self.base_simulator.MOMENTUM_FACTOR = self.parameters['MOMENTUM_FACTOR']
        
        # Configure other parameters as needed
        pass
    
    def _run_simulations(self, num_simulations):
        """
        Run multiple tournament simulations.
        
        Args:
            num_simulations: Number of simulations to run
            
        Returns:
            List of simulation results
        """
        # If we have a simulator, use it
        if self.base_simulator and hasattr(self.base_simulator, 'simulate_tournament'):
            return self.base_simulator.simulate_tournament(num_simulations)
        
        # Otherwise, generate mock simulation results
        return self._generate_mock_simulations(num_simulations)
    
    def _generate_mock_simulations(self, num_simulations):
        """Generate mock simulation results for testing"""
        print("Generating mock simulation results...")
        
        simulations = []
        
        for i in range(num_simulations):
            # Create a simulated tournament result
            simulation = {
                'rounds': {
                    'round_1': [],
                    'round_2': [],
                    'round_3': [],
                    'round_4': [],
                    'round_5': [],
                    'round_6': []
                },
                'champion': None
            }
            
            # For simplicity, we'll just generate a champion
            regions = ["West", "East", "South", "Midwest"]
            seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            
            # Weight more towards lower seeds
            seed_weights = [0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03, 
                          0.02, 0.02, 0.015, 0.01, 0.005, 0.003, 0.002, 0.001]
            
            champion_seed = random.choices(seeds, weights=seed_weights)[0]
            champion_region = random.choice(regions)
            
            simulation['champion'] = {
                'team_id': f"{champion_seed}_{champion_region}",
                'team_name': f"Team {champion_seed}",
                'seed': champion_seed,
                'region': champion_region
            }
            
            simulations.append(simulation)
        
        return simulations
    
    def _generate_consensus_bracket(self, simulation_results):
        """
        Generate a consensus bracket from simulation results.
        
        Args:
            simulation_results: List of simulation results
            
        Returns:
            Consensus bracket
        """
        # If we have the simulator method, use it
        if hasattr(self.base_simulator, 'generate_consensus_bracket'):
            return self.base_simulator.generate_consensus_bracket(simulation_results)
        
        # Otherwise, create a simple consensus based on champion frequency
        champion_counts = defaultdict(int)
        
        for sim in simulation_results:
            if sim['champion']:
                champion_key = f"{sim['champion']['seed']}_{sim['champion']['region']}"
                champion_counts[champion_key] += 1
        
        most_common = max(champion_counts.items(), key=lambda x: x[1])
        seed, region = most_common[0].split('_')
        
        consensus = {
            'champion': {
                'team_id': most_common[0],
                'team_name': f"Team {seed}",
                'seed': int(seed),
                'region': region
            },
            'rounds': {
                'round_1': [],
                'round_2': [],
                'round_3': [],
                'round_4': [],
                'round_5': [],
                'round_6': []
            }
        }
        
        return consensus
    
    def _evaluate_bracket(self, predicted_bracket, actual_tournament):
        """
        Evaluate a predicted bracket against actual tournament results.
        
        Args:
            predicted_bracket: The bracket predicted by the model
            actual_tournament: The actual tournament results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate bracket score
        bracket_score = self._calculate_bracket_score(predicted_bracket, actual_tournament)
        
        # Calculate accuracy by round
        accuracy = self._calculate_prediction_accuracy(predicted_bracket, actual_tournament)
        
        # Check Final Four predictions
        final_four_accuracy = self._calculate_final_four_accuracy(predicted_bracket, actual_tournament)
        
        # Check champion prediction
        champion_accuracy = self._calculate_champion_accuracy(predicted_bracket, actual_tournament)
        
        # Calculate upset detection rate
        upset_detection_rate = self._calculate_upset_detection(predicted_bracket, actual_tournament)
        
        return {
            'bracket_score': bracket_score,
            'accuracy': accuracy,
            'final_four_accuracy': final_four_accuracy,
            'champion_accuracy': champion_accuracy,
            'upset_detection_rate': upset_detection_rate
        }
    
    def _calculate_bracket_score(self, predicted_bracket, actual_tournament):
        """
        Calculate bracket score using standard scoring rules.
        
        Args:
            predicted_bracket: Predicted bracket
            actual_tournament: Actual tournament results
            
        Returns:
            Total bracket score
        """
        # Standard March Madness scoring:
        # Round 1: 1 point per correct pick
        # Round 2: 2 points per correct pick
        # Round 3 (Sweet 16): 4 points per correct pick
        # Round 4 (Elite 8): 8 points per correct pick
        # Round 5 (Final Four): 16 points per correct pick
        # Round 6 (Championship): 32 points for correct champion
        
        round_points = {
            'round_1': 1,
            'round_2': 2,
            'round_3': 4,
            'round_4': 8,
            'round_5': 16,
            'round_6': 32
        }
        
        total_score = 0
        
        # In a real implementation, we would compare each game prediction
        # For this sample, we'll just evaluate the Final Four and Champion
        
        # Check Final Four
        actual_final_four = {f"{team['seed']}_{team['region']}" for team in actual_tournament['final_four']}
        predicted_final_four = set()
        
        if 'rounds' in predicted_bracket and 'round_5' in predicted_bracket['rounds']:
            for game in predicted_bracket['rounds']['round_5']:
                if 'team1' in game:
                    predicted_final_four.add(f"{game['team1']['seed']}_{game['team1']['region']}")
                if 'team2' in game:
                    predicted_final_four.add(f"{game['team2']['seed']}_{game['team2']['region']}")
        
        # Score points for each correct Final Four team
        for team in predicted_final_four.intersection(actual_final_four):
            total_score += round_points['round_5']
        
        # Check champion
        if predicted_bracket['champion'] and actual_tournament['champion']:
            predicted_champion = f"{predicted_bracket['champion']['seed']}_{predicted_bracket['champion']['region']}"
            actual_champion = f"{actual_tournament['champion']['seed']}_{actual_tournament['champion']['region']}"
            
            if predicted_champion == actual_champion:
                total_score += round_points['round_6']
        
        return total_score
    
    def _calculate_prediction_accuracy(self, predicted_bracket, actual_tournament):
        """
        Calculate overall prediction accuracy.
        
        Args:
            predicted_bracket: Predicted bracket
            actual_tournament: Actual tournament results
            
        Returns:
            Percentage of correct predictions
        """
        # In a real implementation, this would compare game-by-game
        # For this sample, we'll return a simple estimate
        return 70.0  # 70% accuracy
    
    def _calculate_final_four_accuracy(self, predicted_bracket, actual_tournament):
        """Calculate Final Four prediction accuracy"""
        actual_final_four = {f"{team['seed']}_{team['region']}" for team in actual_tournament['final_four']}
        predicted_final_four = set()
        
        if 'rounds' in predicted_bracket and 'round_5' in predicted_bracket['rounds']:
            for game in predicted_bracket['rounds']['round_5']:
                if 'team1' in game:
                    predicted_final_four.add(f"{game['team1']['seed']}_{game['team1']['region']}")
                if 'team2' in game:
                    predicted_final_four.add(f"{game['team2']['seed']}_{game['team2']['region']}")
        
        # Calculate accuracy
        num_correct = len(predicted_final_four.intersection(actual_final_four))
        accuracy = (num_correct / 4) * 100 if predicted_final_four else 0
        
        return accuracy
    
    def _calculate_champion_accuracy(self, predicted_bracket, actual_tournament):
        """Calculate champion prediction accuracy"""
        if not predicted_bracket['champion'] or not actual_tournament['champion']:
            return 0
        
        predicted_champion = f"{predicted_bracket['champion']['seed']}_{predicted_bracket['champion']['region']}"
        actual_champion = f"{actual_tournament['champion']['seed']}_{actual_tournament['champion']['region']}"
        
        return 100 if predicted_champion == actual_champion else 0
    
    def _calculate_upset_detection(self, predicted_bracket, actual_tournament):
        """Calculate rate of detecting upsets"""
        if 'upsets' not in actual_tournament:
            return 0
        
        # In a real implementation, we would check each upset prediction
        # For this sample, we'll return an estimate
        return 50.0  # 50% of upsets predicted
    
    def grid_search(self, parameter_grid):
        """
        Perform grid search across multiple parameters.
        
        Args:
            parameter_grid: Dictionary mapping parameters to lists of values to test
            
        Returns:
            Best parameter combination and its performance
        """
        print(f"Performing grid search across {len(parameter_grid)} parameters...")
        
        # Save original parameters
        original_parameters = self.parameters.copy()
        
        # Generate all combinations of parameters
        import itertools
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        best_score = -1
        best_params = None
        results = []
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for i, combination in enumerate(combinations):
            # Set parameters for this combination
            for name, value in zip(param_names, combination):
                if "." in name:
                    main_param, sub_param = name.split(".")
                    self.parameters[main_param][int(sub_param)] = value
                else:
                    self.parameters[name] = value
            
            # Run backtests
            metrics = self.run_backtests(num_simulations=50)  # Reduced simulations for grid search
            
            # Store results
            result = {
                'parameters': {name: value for name, value in zip(param_names, combination)},
                'bracket_score': metrics['average_bracket_score'],
                'accuracy': metrics['average_accuracy'],
                'final_four_accuracy': metrics['final_four_accuracy'],
                'champion_accuracy': metrics['champion_accuracy']
            }
            results.append(result)
            
            # Check if this is the best so far
            if metrics['average_bracket_score'] > best_score:
                best_score = metrics['average_bracket_score']
                best_params = {name: value for name, value in zip(param_names, combination)}
            
            print(f"Combination {i+1}/{len(combinations)} complete")
        
        # Reset parameters
        self.parameters = original_parameters
        
        print(f"Grid search complete!")
        print(f"Best parameters: {best_params}")
        print(f"Best bracket score: {best_score:.2f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results
        }
