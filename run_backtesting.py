# run_backtesting.py - Main Execution Script Using ESPN Data
import json
import argparse
from simulator import MarchMadnessSimulator
from backtesting_framework import MarchMadnessBacktester

def parse_arguments():
    parser = argparse.ArgumentParser(description='March Madness Backtesting Runner')
    parser.add_argument('--years', type=str, help='Years to include in backtesting')
    parser.add_argument('--sims', type=int, default=500, help='Number of simulations per backtest')
    return parser.parse_args()

def main():
    args = parse_arguments()
    years = [int(y.strip()) for y in args.years.split(',')] if args.years else list(range(2015, 2023))
    
    print("Initializing March Madness Simulator...")
    simulator = MarchMadnessSimulator(use_real_data=True)
    
    print("Running Backtesting...")
    backtester = MarchMadnessBacktester(simulator, years)
    backtester.load_historical_tournaments()
    results = backtester.run_backtests(num_simulations=args.sims)
    
    with open('final_backtesting_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Backtesting Complete. Results saved to final_backtesting_results.json")
    
    print("Running Final Tournament Simulation with Optimized Parameters...")
    final_predictions = simulator.simulate_tournament(10000)
    
    with open('final_consensus_bracket.json', 'w') as f:
        json.dump(final_predictions, f, indent=4)
    
    print("Final March Madness Predictions Saved.")

if __name__ == "__main__":
    main()
