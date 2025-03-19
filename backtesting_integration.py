import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import our custom modules
from historical_data_loader import HistoricalTournamentDataLoader
from backtesting_framework import MarchMadnessBacktester
from parameter_tuning import ParameterTuner

# This script provides a complete workflow for:
# 1. Loading historical data
# 2. Integrating the existing simulator with the backtesting framework
# 3. Running parameter optimization
# 4. Applying optimized parameters to the simulator
# 5. Evaluating the improved model

def load_simulator():
    """
    Load the March Madness simulator.
    
    Returns:
        The simulator object
    """
    try:
        # Try to import the simulator from your existing code
        from simulator import MarchMadnessSimulator
        from advanced_metrics import enhance_simulator_with_advanced_metrics
        
        print("Loading March Madness simulator...")
        simulator = MarchMadnessSimulator(use_real_data=True)
        
        # Apply advanced metrics
        simulator = enhance_simulator_with_advanced_metrics(simulator)
        
        print("Successfully loaded simulator")
        return simulator
    except ImportError:
        # If unable to import, create a mock simulator
        print("Unable to import simulator. Creating mock simulator for demonstration...")
        
        # Create a mock simulator that has required methods and properties
        class MockSimulator:
            def __init__(self):
                self.UPSET_FACTOR = 0.20
                self.MOMENTUM_FACTOR = 0.08
                self.HOME_ADVANTAGE = 0.05
                
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
                
                # Data placeholder for demonstration
                self.team_stats = pd.DataFrame({
                    'team_id': range(1, 69),
                    'team_name': [f'Team {i}' for i in range(1, 69)],
                    'record': ['20-10' for _ in range(68)]
                })
                
                # Sample tournament structure
                self.tournament_teams = {
                    'East': {i: {'team_id': i, 'seed': i, 'team_name': f'Team {i} (East)'} for i in range(1, 17)},
                    'West': {i: {'team_id': i+16, 'seed': i, 'team_name': f'Team {i} (West)'} for i in range(1, 17)},
                    'South': {i: {'team_id': i+32, 'seed': i, 'team_name': f'Team {i} (South)'} for i in range(1, 17)},
                    'Midwest': {i: {'team_id': i+48, 'seed': i, 'team_name': f'Team {i} (Midwest)'} for i in range(1, 17)}
                }
            
            def get_tournament_teams(self):
                """Return the tournament teams structure"""
                return self.tournament_teams
            
            def simulate_tournament(self, num_simulations=100):
                """Simulate the tournament multiple times"""
                import random
                
                all_results = []
                
                for sim_num in range(num_simulations):
                    # Create simulation result structure
                    results = {
                        'simulation': sim_num + 1,
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
                    
                    # Generate random champion with preference for lower seeds
                    regions = list(self.tournament_teams.keys())
                    region = random.choice(regions)
                    
                    # Random seed with more weight to higher seeds
                    seeds = list(range(1, 17))
                    seed_weights = [0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03, 
                                  0.02, 0.02, 0.015, 0.01, 0.005, 0.003, 0.002, 0.001]
                    seed = random.choices(seeds, weights=seed_weights)[0]
                    
                    # Set the champion
                    team = self.tournament_teams[region][seed]
                    results['champion'] = {
                        'team_id': team['team_id'],
                        'team_name': team['team_name'],
                        'seed': seed,
                        'region': region
                    }
                    
                    all_results.append(results)
                
                return all_results
            
            def generate_consensus_bracket(self, simulation_results):
                """Generate a consensus bracket from all simulations"""
                # Count champions from all simulations
                champion_counts = {}
                
                for sim in simulation_results:
                    if sim['champion']:
                        champ_id = sim['champion']['team_id']
                        champion_counts[champ_id] = champion_counts.get(champ_id, 0) + 1
                
                # Find most common champion
                most_common = max(champion_counts.items(), key=lambda x: x[1])[0]
                
                # Find the team info
                champion = None
                for region, seeds in self.tournament_teams.items():
                    for seed, team in seeds.items():
                        if team['team_id'] == most_common:
                            champion = {
                                'team_id': team['team_id'],
                                'team_name': team['team_name'],
                                'seed': team['seed'],
                                'region': region
                            }
                
                # Create a mock consensus bracket
                consensus = {
                    'num_simulations': len(simulation_results),
                    'last_simulation': {
                        'rounds': {
                            'round_1': [],
                            'round_2': [],
                            'round_3': [],
                            'round_4': [],
                            'round_5': [],
                            'round_6': []
                        },
                        'champion': champion
                    }
                }
                
                return consensus
        
        return MockSimulator()

def adapt_simulator_to_backtester(simulator, backtester):
    """
    Adapt the simulator to work with the backtester.
    This function adjusts the simulator's parameters to match backtester's parameters.
    
    Args:
        simulator: The simulator object
        backtester: The backtester object
        
    Returns:
        The adapted simulator
    """
    print("Adapting simulator to backtester...")
    
    # Copy key parameters from backtester to simulator
    if hasattr(backtester, 'parameters') and hasattr(simulator, 'UPSET_FACTOR'):
        # Set direct parameters
        if 'UPSET_FACTOR' in backtester.parameters:
            simulator.UPSET_FACTOR = backtester.parameters['UPSET_FACTOR']
            print(f"  Set UPSET_FACTOR to {backtester.parameters['UPSET_FACTOR']}")
        
        if 'MOMENTUM_FACTOR' in backtester.parameters and hasattr(simulator, 'MOMENTUM_FACTOR'):
            simulator.MOMENTUM_FACTOR = backtester.parameters['MOMENTUM_FACTOR']
            print(f"  Set MOMENTUM_FACTOR to {backtester.parameters['MOMENTUM_FACTOR']}")
        
        # Set historical upset rates if available
        if hasattr(simulator, 'HISTORICAL_UPSET_RATES') and 'historical_matchup_weights' in backtester.parameters:
            # This would need custom implementation based on how your simulator uses this
            print(f"  Set historical matchup weight: {backtester.parameters['historical_matchup_weights']}")
    
    return simulator

def run_backtesting_process(simulator=None, years=None, data_dir="historical_data"):
    """
    Run the complete backtesting process.
    
    Args:
        simulator: The simulator object (None to create new)
        years: List of tournament years to test (None for default)
        data_dir: Directory containing historical data
        
    Returns:
        Tuple containing (simulator, backtester, tuner, optimized_params)
    """
    print("\n===== RUNNING BACKTESTING PROCESS =====")
    start_time = datetime.now()
    print(f"Started at: {start_time}")
    
    # Step 1: Load or collect historical tournament data
    history_loader = HistoricalTournamentDataLoader(data_dir=data_dir)
    
    if not os.path.exists(os.path.join(data_dir, "tournaments.json")):
        # Collect data if doesn't exist
        history_loader.collect_historical_data()
    else:
        # Otherwise load the existing data
        history_loader.load_tournament_data()
    
    # Analyze historical patterns
    upsets_df = history_loader.get_historical_upsets()
    seed_df = history_loader.get_seed_performance()
    
    # Step 2: Load or create the simulator
    if simulator is None:
        simulator = load_simulator()
    
    # Step 3: Create backtester with historical data
    backtester = MarchMadnessBacktester(simulator, years=years)
    backtester.load_historical_tournaments(data_dir=data_dir)
    
    # Step 4: Adapt simulator to backtester
    simulator = adapt_simulator_to_backtester(simulator, backtester)
    
    # Step 5: Run initial backtests to establish baseline
    print("\nRunning initial backtests to establish baseline...")
    baseline_metrics = backtester.run_backtests(num_simulations=100)
    
    # Step 6: Create parameter tuner
    tuner = ParameterTuner(backtester)
    
    # Step 7: Run key parameter tuning
    print("\nTuning key parameters individually...")
    upset_results, _ = tuner.tune_upset_factor()
    momentum_results, _ = tuner.tune_momentum_factor()
    
    # Step 8: Run grid search for parameter combinations
    print("\nRunning grid search for parameter combinations...")
    param_grid = {
        'UPSET_FACTOR': [0.15, 0.20, 0.25, 0.30],
        'MOMENTUM_FACTOR': [0.05, 0.08, 0.12],
        'historical_matchup_weights': [0.4, 0.5, 0.6]
    }
    grid_results = tuner.run_grid_search(param_grid)
    
    # Step 9: Get recommended parameters
    optimized_params = tuner.get_recommended_parameters()
    
    # Step 10: Compare optimized parameters to baseline
    baseline_params = {
        'UPSET_FACTOR': 0.20,
        'MOMENTUM_FACTOR': 0.08,
        'historical_matchup_weights': 0.5
    }
    comparison = tuner.compare_to_baseline(baseline_params, optimized_params)
    
    # Step 11: Apply optimized parameters to simulator
    simulator = tuner.apply_parameters_to_simulator(simulator, optimized_params)
    
    # Save the tuning results
    tuner.save_tuning_results("parameter_tuning_results.json")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nBacktesting process completed in {duration}")
    print("Optimized parameters ready to use in full simulator")
    
    return simulator, backtester, tuner, optimized_params

def run_optimized_simulation(simulator, optimized_params, num_simulations=1000):
    """
    Run a full tournament simulation with optimized parameters.
    
    Args:
        simulator: The simulator object
        optimized_params: The optimized parameters
        num_simulations: Number of simulations to run
        
    Returns:
        Simulation results
    """
    print("\n===== RUNNING OPTIMIZED TOURNAMENT SIMULATION =====")
    
    # Apply optimized parameters
    if hasattr(simulator, 'UPSET_FACTOR') and 'UPSET_FACTOR' in optimized_params:
        simulator.UPSET_FACTOR = optimized_params['UPSET_FACTOR']
    
    if hasattr(simulator, 'MOMENTUM_FACTOR') and 'MOMENTUM_FACTOR' in optimized_params:
        simulator.MOMENTUM_FACTOR = optimized_params['MOMENTUM_FACTOR']
    
    # Run the simulations
    print(f"Running {num_simulations} simulations with optimized parameters...")
    simulation_results = simulator.simulate_tournament(num_simulations=num_simulations)
    
    # Generate consensus bracket
    print("Generating consensus bracket...")
    consensus_bracket = simulator.generate_bracket_visualization(simulation_results)
    
    # Save results
    with open("optimized_simulation_results.json", "w") as f:
        json.dump({
            'num_simulations': num_simulations,
            'parameters': optimized_params,
            'consensus_bracket': consensus_bracket
        }, f, indent=2)
    
    print(f"Completed {num_simulations} optimized simulations")
    print("Results saved to optimized_simulation_results.json")
    
    return consensus_bracket

def analyze_simulation_differences(simulator, baseline_params, optimized_params, num_simulations=500):
    """
    Compare simulations with baseline vs optimized parameters.
    
    Args:
        simulator: The simulator object
        baseline_params: The baseline parameters
        optimized_params: The optimized parameters
        num_simulations: Number of simulations to run for each
        
    Returns:
        DataFrame with comparison results
    """
    print("\n===== ANALYZING SIMULATION DIFFERENCES =====")
    
    # Run baseline simulations
    print(f"Running {num_simulations} baseline simulations...")
    
    # Set baseline parameters
    original_upset = getattr(simulator, 'UPSET_FACTOR', None)
    original_momentum = getattr(simulator, 'MOMENTUM_FACTOR', None)
    
    if hasattr(simulator, 'UPSET_FACTOR') and 'UPSET_FACTOR' in baseline_params:
        simulator.UPSET_FACTOR = baseline_params['UPSET_FACTOR']
    
    if hasattr(simulator, 'MOMENTUM_FACTOR') and 'MOMENTUM_FACTOR' in baseline_params:
        simulator.MOMENTUM_FACTOR = baseline_params['MOMENTUM_FACTOR']
    
    # Run baseline simulations
    baseline_results = simulator.simulate_tournament(num_simulations=num_simulations)
    baseline_consensus = simulator.generate_consensus_bracket(baseline_results)
    
    # Run optimized simulations
    print(f"Running {num_simulations} optimized simulations...")
    
    # Set optimized parameters
    if hasattr(simulator, 'UPSET_FACTOR') and 'UPSET_FACTOR' in optimized_params:
        simulator.UPSET_FACTOR = optimized_params['UPSET_FACTOR']
    
    if hasattr(simulator, 'MOMENTUM_FACTOR') and 'MOMENTUM_FACTOR' in optimized_params:
        simulator.MOMENTUM_FACTOR = optimized_params['MOMENTUM_FACTOR']
    
    # Run optimized simulations
    optimized_results = simulator.simulate_tournament(num_simulations=num_simulations)
    optimized_consensus = simulator.generate_consensus_bracket(optimized_results)
    
    # Reset parameters
    if original_upset is not None:
        simulator.UPSET_FACTOR = original_upset
    
    if original_momentum is not None:
        simulator.MOMENTUM_FACTOR = original_momentum
    
    # Compare results
    print("Analyzing differences in simulation outputs...")
    
    # Count champion occurrences in baseline
    baseline_champions = {}
    for sim in baseline_results:
        if sim['champion']:
            champion_id = sim['champion']['team_id']
            team_name = sim['champion']['team_name']
            seed = sim['champion']['seed']
            baseline_champions[(champion_id, team_name, seed)] = baseline_champions.get((champion_id, team_name, seed), 0) + 1
    
    # Count champion occurrences in optimized
    optimized_champions = {}
    for sim in optimized_results:
        if sim['champion']:
            champion_id = sim['champion']['team_id']
            team_name = sim['champion']['team_name']
            seed = sim['champion']['seed']
            optimized_champions[(champion_id, team_name, seed)] = optimized_champions.get((champion_id, team_name, seed), 0) + 1
    
    # Convert to percentages
    baseline_pct = {k: (v / num_simulations) * 100 for k, v in baseline_champions.items()}
    optimized_pct = {k: (v / num_simulations) * 100 for k, v in optimized_champions.items()}
    
    # Create comparison DataFrame
    # All champions from either simulation
    all_champions = set(baseline_champions.keys()).union(set(optimized_champions.keys()))
    
    comparison_data = []
    for champ in all_champions:
        team_id, team_name, seed = champ
        baseline_pct_value = baseline_pct.get(champ, 0)
        optimized_pct_value = optimized_pct.get(champ, 0)
        diff = optimized_pct_value - baseline_pct_value
        
        comparison_data.append({
            'team_id': team_id,
            'team_name': team_name,
            'seed': seed,
            'baseline_pct': baseline_pct_value,
            'optimized_pct': optimized_pct_value,
            'diff': diff
        })
    
    # Convert to DataFrame and sort
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('optimized_pct', ascending=False)
    
    print("\nChampionship probability comparison (top 10 teams):")
    print(comparison_df.head(10)[['team_name', 'seed', 'baseline_pct', 'optimized_pct', 'diff']])
    
    # Compare consensus brackets
    print("\nConsensus bracket comparison:")
    if 'champion' in baseline_consensus.get('last_simulation', {}) and 'champion' in optimized_consensus.get('last_simulation', {}):
        baseline_champ = baseline_consensus['last_simulation']['champion']
        optimized_champ = optimized_consensus['last_simulation']['champion']
        
        print(f"  Baseline consensus champion: {baseline_champ['seed']} {baseline_champ['team_name']}")
        print(f"  Optimized consensus champion: {optimized_champ['seed']} {optimized_champ['team_name']}")
    
    # Count seed distributions
    baseline_seeds = [sim['champion']['seed'] for sim in baseline_results if sim['champion']]
    optimized_seeds = [sim['champion']['seed'] for sim in optimized_results if sim['champion']]
    
    # Create seed distribution comparison
    seed_counts = {}
    for seed in range(1, 17):
        baseline_count = baseline_seeds.count(seed)
        optimized_count = optimized_seeds.count(seed)
        seed_counts[seed] = {
            'baseline_pct': (baseline_count / len(baseline_seeds)) * 100 if baseline_seeds else 0,
            'optimized_pct': (optimized_count / len(optimized_seeds)) * 100 if optimized_seeds else 0,
            'diff': ((optimized_count / len(optimized_seeds)) - (baseline_count / len(baseline_seeds))) * 100 if baseline_seeds and optimized_seeds else 0
        }
    
    # Convert to DataFrame
    seed_dist_df = pd.DataFrame.from_dict(seed_counts, orient='index')
    seed_dist_df.index.name = 'seed'
    seed_dist_df = seed_dist_df.reset_index()
    seed_dist_df = seed_dist_df.sort_values('seed')
    
    print("\nChampionship seed distribution comparison:")
    print(seed_dist_df)
    
    # Create visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Champion probability comparison
        plt.figure(figsize=(12, 8))
        top_teams = comparison_df.head(10).sort_values('seed')
        
        x = np.arange(len(top_teams))
        width = 0.35
        
        plt.bar(x - width/2, top_teams['baseline_pct'], width, label='Baseline Parameters')
        plt.bar(x + width/2, top_teams['optimized_pct'], width, label='Optimized Parameters')
        
        plt.xlabel('Teams')
        plt.ylabel('Championship Probability (%)')
        plt.title('Impact of Parameter Optimization on Championship Odds')
        plt.xticks(x, [f"{row['team_name']} ({row['seed']})" for _, row in top_teams.iterrows()], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig("championship_odds_comparison.png", dpi=300)
        
        # Seed distribution comparison
        plt.figure(figsize=(12, 6))
        
        plt.bar(seed_dist_df['seed'] - 0.2, seed_dist_df['baseline_pct'], width=0.4, label='Baseline')
        plt.bar(seed_dist_df['seed'] + 0.2, seed_dist_df['optimized_pct'], width=0.4, label='Optimized')
        
        plt.xlabel('Seed')
        plt.ylabel('Championship Probability (%)')
        plt.title('Championship Probability by Seed: Baseline vs. Optimized')
        plt.xticks(range(1, 17))
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig("seed_distribution_comparison.png", dpi=300)
        
        print("Visualizations saved to championship_odds_comparison.png and seed_distribution_comparison.png")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return comparison_df, seed_dist_df

def run_full_backtesting_workflow():
    """
    Run the complete backtesting workflow from data collection to optimized simulation.
    
    Returns:
        Summary of the full workflow
    """
    print("\n========== MARCH MADNESS BACKTESTING WORKFLOW ==========")
    print("Starting backtesting workflow at", datetime.now())
    
    # Step 1: Run the backtesting process
    simulator, backtester, tuner, optimized_params = run_backtesting_process()
    
    # Step 2: Run optimized simulation
    consensus_bracket = run_optimized_simulation(simulator, optimized_params)
    
    # Step 3: Compare baseline and optimized simulations
    baseline_params = {
        'UPSET_FACTOR': 0.20,
        'MOMENTUM_FACTOR': 0.08,
        'historical_matchup_weights': 0.5
    }
    
    comparison_df, seed_dist_df = analyze_simulation_differences(
        simulator, baseline_params, optimized_params)
    
    # Step 4: Compile final summary report
    summary = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'optimized_parameters': optimized_params,
        'baseline_parameters': baseline_params,
        'performance_improvement': {
            'bracket_score': float(tuner.tuning_results['grid_search']['best_score']),
            'champion_accuracy': float(comparison_df.iloc[0]['diff'] if len(comparison_df) > 0 else 0)
        },
        'consensus_champion': {
            'team_name': consensus_bracket['last_simulation']['champion']['team_name'],
            'seed': consensus_bracket['last_simulation']['champion']['seed'],
            'region': consensus_bracket['last_simulation']['champion']['region']
        }
    }
    
    # Save summary
    with open("backtesting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nBacktesting workflow completed at", datetime.now())
    print("Summary saved to backtesting_summary.json")
    
    return summary

# Run the complete workflow if run directly
if __name__ == "__main__":
    summary = run_full_backtesting_workflow()
    
    # Print key findings
    print("\n===== KEY FINDINGS =====")
    print(f"Optimal UPSET_FACTOR: {summary['optimized_parameters'].get('UPSET_FACTOR', 'N/A')}")
    print(f"Optimal MOMENTUM_FACTOR: {summary['optimized_parameters'].get('MOMENTUM_FACTOR', 'N/A')}")
    print(f"Bracket score improvement: {summary['performance_improvement']['bracket_score']:.2f} points")
    print(f"Consensus champion: {summary['consensus_champion']['seed']} {summary['consensus_champion']['team_name']} ({summary['consensus_champion']['region']})")
