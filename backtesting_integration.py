# backtesting_integration.py - Full Integration of Backtesting and Simulation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import necessary modules
from historical_data_loader import HistoricalTournamentDataLoader
from backtesting_framework import MarchMadnessBacktester
from parameter_tuning import ParameterTuner
from simulator import MarchMadnessSimulator

def adapt_simulator_to_backtester(simulator, backtester):
    """Ensure the simulator uses the best parameters from backtesting."""
    print("Adapting simulator with optimized parameters...")
    if hasattr(backtester, 'parameters'):
        simulator.UPSET_FACTOR = backtester.parameters.get('UPSET_FACTOR', simulator.UPSET_FACTOR)
        simulator.MOMENTUM_FACTOR = backtester.parameters.get('MOMENTUM_FACTOR', simulator.MOMENTUM_FACTOR)
    return simulator

def run_backtesting_process(simulator=None, years=None, data_dir="historical_data", verbose=False):
    """Run the full backtesting and optimization process."""
    print("Running backtesting process...")
    
    history_loader = HistoricalTournamentDataLoader(data_dir=data_dir)
    history_loader.load_tournament_data()
    
    if simulator is None:
        simulator = MarchMadnessSimulator(use_real_data=True)
    
    backtester = MarchMadnessBacktester(simulator, years=years)
    backtester.load_historical_tournaments(data_dir=data_dir)
    
    print("Running backtesting...")
    backtesting_results = backtester.run_backtests(num_simulations=500)
    
    tuner = ParameterTuner(backtester)
    optimized_params = tuner.run_optimization()
    
    simulator = adapt_simulator_to_backtester(simulator, backtester)
    
    print("Saving optimized parameters...")
    with open("optimized_parameters.json", "w") as f:
        json.dump(optimized_params, f, indent=4)
    
    return simulator, backtester, tuner, optimized_params

def run_optimized_simulation(simulator, optimized_params, num_simulations=10000):
    """Run the final tournament simulation using optimized parameters."""
    print("Running optimized tournament simulation...")
    
    simulator.UPSET_FACTOR = optimized_params.get("UPSET_FACTOR", simulator.UPSET_FACTOR)
    simulator.MOMENTUM_FACTOR = optimized_params.get("MOMENTUM_FACTOR", simulator.MOMENTUM_FACTOR)
    
    final_predictions = simulator.simulate_tournament(num_simulations)
    
    print("Saving final consensus bracket...")
    with open("final_consensus_bracket.json", "w") as f:
        json.dump(final_predictions, f, indent=4)
    
    return final_predictions

if __name__ == "__main__":
    years = list(range(2015, 2023))
    simulator, backtester, tuner, optimized_params = run_backtesting_process(years=years)
    run_optimized_simulation(simulator, optimized_params)
