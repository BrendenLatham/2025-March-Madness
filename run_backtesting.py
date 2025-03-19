#!/usr/bin/env python3
"""
March Madness Backtesting Runner

This script executes a complete backtesting workflow to optimize
the March Madness prediction model based on historical tournament results.

Usage:
  python run_backtesting.py [--years YEARS] [--sims SIMS] [--datadir DIR]

Options:
  --years YEARS   Years to include in backtesting (e.g., "2015,2016,2017,2018,2019")
  --sims SIMS     Number of simulations to run in each test [default: 200]
  --datadir DIR   Directory containing historical data [default: "historical_data"]
"""

import os
import sys
import json
import argparse
from datetime import datetime

parser.add_argument('--quiet', action='store_true',
                  help='Suppress detailed output during simulations')

# Import your actual simulator
try:
    # Try to import the simulator directly
    from simulator import MarchMadnessSimulator
    from advanced_metrics import enhance_simulator_with_advanced_metrics
    SIMULATOR_AVAILABLE = True
except ImportError:
    # If not available as a module, look for the file
    if os.path.exists("simulator.py"):
        # Import using importlib
        import importlib.util
        spec = importlib.util.spec_from_file_location("simulator", "simulator.py")
        simulator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(simulator_module)
        MarchMadnessSimulator = simulator_module.MarchMadnessSimulator
        
        # Import advanced metrics
        spec2 = importlib.util.spec_from_file_location("advanced_metrics", "advanced_metrics.py")
        metrics_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(metrics_module)
        enhance_simulator_with_advanced_metrics = metrics_module.enhance_simulator_with_advanced_metrics
        
        SIMULATOR_AVAILABLE = True
    else:
        print("Warning: Could not find simulator.py, will use mock simulator")
        SIMULATOR_AVAILABLE = False

# Import backtesting modules
from historical_data_loader import HistoricalTournamentDataLoader
from backtesting_framework import MarchMadnessBacktester
from parameter_tuning import ParameterTuner
from backtesting_integration import run_backtesting_process, run_optimized_simulation, analyze_simulation_differences

def parse_arguments():
    parser = argparse.ArgumentParser(description='March Madness Backtesting Runner')
    
    parser.add_argument('--years', type=str, 
                      help='Years to include in backtesting (comma-separated, e.g., "2015,2016,2017,2018,2019")')
    
    parser.add_argument('--sims', type=int, default=200,
                      help='Number of simulations to run in each test')
    
    parser.add_argument('--datadir', type=str, default="historical_data",
                      help='Directory containing historical data')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Process years argument
    years = None
    if args.years:
        years = [int(y.strip()) for y in args.years.split(',')]
    
    print("=== MARCH MADNESS BACKTESTING ===")
    print(f"Started at: {datetime.now()}")
    
    # Check if data directory exists, create if it doesn't
    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)
        print(f"Created data directory: {args.datadir}")
    
    # Step 1: Initialize the simulator
    if SIMULATOR_AVAILABLE:
        print("Initializing March Madness simulator...")
        simulator = MarchMadnessSimulator(use_real_data=True)
        simulator = enhance_simulator_with_advanced_metrics(simulator)
    else:
        print("March Madness simulator not available.")
        print("Creating mock simulator for demonstration...")
        simulator = None  # Will use mock simulator in run_backtesting_process()
    
    # Step 2: Run the backtesting process
    print("\nRunning backtesting process...")
    simulator, backtester, tuner, optimized_params = run_backtesting_process(
        simulator=simulator, 
        years=years,
        data_dir=args.datadir
    )
    
    # Step 3: Run optimized simulation
    print("\nRunning tournament simulation with optimized parameters...")
    consensus_bracket = run_optimized_simulation(
        simulator=simulator,
        optimized_params=optimized_params,
        num_simulations=args.sims
    )
    
    # Step 4: Compare baseline and optimized simulations
    print("\nComparing baseline and optimized simulation results...")
    baseline_params = {
        'UPSET_FACTOR': 0.20,
        'MOMENTUM_FACTOR': 0.08,
        'historical_matchup_weights': 0.5
    }
    
    comparison_df, seed_dist_df = analyze_simulation_differences(
        simulator=simulator,
        baseline_params=baseline_params,
        optimized_params=optimized_params,
        num_simulations=args.sims
    )
    
    # Step 5: Generate final report
    print("\nGenerating final report...")
    
    report = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "years_tested": years if years else "all available",
        "simulations_per_test": args.sims,
        "baseline_parameters": baseline_params,
        "optimized_parameters": optimized_params,
        "performance_improvement": {
            "bracket_score": float(tuner.tuning_results.get('grid_search', {}).get('best_score', 0)),
            "champion_selection_accuracy": float(comparison_df.iloc[0]['diff'] if len(comparison_df) > 0 else 0)
        },
        "consensus_champion": {
            "team_name": consensus_bracket['last_simulation']['champion']['team_name'],
            "seed": consensus_bracket['last_simulation']['champion']['seed'],
            "region": consensus_bracket['last_simulation']['champion']['region']
        },
        "top_championship_contenders": comparison_df.head(10)[['team_name', 'seed', 'optimized_pct']].to_dict('records'),
        "seed_distribution": {str(row['seed']): row['optimized_pct'] for _, row in seed_dist_df.iterrows()}
    }
    
    # Save final report
    with open("backtesting_final_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nBacktesting complete!")
    print(f"Final report saved to backtesting_final_report.json")
    print(f"Ended at: {datetime.now()}")
    
    return report

if __name__ == "__main__":
    main()
