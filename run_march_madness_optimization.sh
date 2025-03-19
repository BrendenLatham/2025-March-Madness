#!/bin/bash
# run_march_madness_optimization.sh
# Complete workflow for March Madness prediction optimization

# Set script to exit on error
set -e

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

echo -e "${BLUE}===============================================${RESET}"
echo -e "${GREEN}MARCH MADNESS PREDICTION OPTIMIZATION WORKFLOW${RESET}"
echo -e "${BLUE}===============================================${RESET}"
echo "Started at: $(date)"
echo

# Create directories if they don't exist
echo -e "${YELLOW}Creating necessary directories...${RESET}"
mkdir -p historical_data
mkdir -p results
mkdir -p visualizations

# Step 1: Data Collection
echo -e "\n${GREEN}STEP 1: COLLECTING CURRENT SEASON DATA${RESET}"
echo -e "${YELLOW}Fetching team info, statistics, and rankings from ESPN...${RESET}"
python data_collection.py

# Step 2: Run Backtesting to optimize parameters
echo -e "\n${GREEN}STEP 2: RUNNING BACKTESTING FOR PARAMETER OPTIMIZATION${RESET}"
echo -e "${YELLOW}Testing against historical tournaments (2015-2023)...${RESET}"
python run_backtesting.py --years 2015,2016,2017,2018,2019,2020,2021,2022,2023 --sims 500 --datadir historical_data

# Step 3: Extract optimized parameters and save to a file
echo -e "\n${GREEN}STEP 3: EXTRACTING OPTIMIZED PARAMETERS${RESET}"
echo -e "${YELLOW}Reading optimal parameters from backtesting report...${RESET}"

# Extract parameters from the backtesting report using jq if available
if command -v jq &> /dev/null; then
    UPSET_FACTOR=$(jq -r '.optimized_parameters.UPSET_FACTOR' backtesting_final_report.json)
    MOMENTUM_FACTOR=$(jq -r '.optimized_parameters.MOMENTUM_FACTOR' backtesting_final_report.json)
    
    # Create a parameters file
    cat > optimized_parameters.json << EOL
{
    "UPSET_FACTOR": $UPSET_FACTOR,
    "MOMENTUM_FACTOR": $MOMENTUM_FACTOR
}
EOL
    echo "Saved optimized parameters to optimized_parameters.json"
    echo -e "${BLUE}Optimized UPSET_FACTOR: ${RESET}$UPSET_FACTOR"
    echo -e "${BLUE}Optimized MOMENTUM_FACTOR: ${RESET}$MOMENTUM_FACTOR"
else
    echo "jq not found. Please manually extract parameters from backtesting_final_report.json"
    echo "You can install jq with: apt-get install jq (Ubuntu/Debian) or brew install jq (macOS)"
fi

# Step 4: Run full tournament simulation with optimized parameters
echo -e "\n${GREEN}STEP 4: RUNNING FULL TOURNAMENT SIMULATION WITH OPTIMIZED PARAMETERS${RESET}"
echo -e "${YELLOW}Simulating 10,000 tournament runs with optimized parameters...${RESET}"

# Create a script to run the optimized simulation
cat > run_optimized_simulation.py << EOL
import json
import os
from simulator import MarchMadnessSimulator
from advanced_metrics import enhance_simulator_with_advanced_metrics

# Load optimized parameters
with open('optimized_parameters.json', 'r') as f:
    params = json.load(f)

# Initialize and enhance simulator
print("Initializing simulator with optimized parameters...")
simulator = MarchMadnessSimulator(use_real_data=True)
simulator = enhance_simulator_with_advanced_metrics(simulator)

# Apply optimized parameters
simulator.UPSET_FACTOR = params['UPSET_FACTOR']
simulator.MOMENTUM_FACTOR = params['MOMENTUM_FACTOR']
print(f"Applied UPSET_FACTOR: {simulator.UPSET_FACTOR}")
print(f"Applied MOMENTUM_FACTOR: {simulator.MOMENTUM_FACTOR}")

# Run simulation
print("Running 10,000 tournament simulations...")
num_simulations = 10000
results = simulator.run_simulation(num_simulations=num_simulations)

# Generate visualizations
print("Generating visualizations...")
simulator.visualize_results(results)

# Generate bracket visualization
bracket_text = simulator.generate_bracket_visualization(results)
with open("optimized_bracket.txt", "w") as f:
    f.write(bracket_text)

# Save detailed results
print("Saving results...")
simulator.save_results(results, "results/optimized_simulation_results.json")

print("Optimized simulation complete!")
print(f"Championship odds chart saved to championship_odds.png")
print(f"Final Four odds chart saved to final_four_odds.png")
print(f"Bracket visualization saved to optimized_bracket.txt")
EOL

# Run the optimized simulation
python run_optimized_simulation.py

# Step 5: Organize outputs
echo -e "\n${GREEN}STEP 5: ORGANIZING OUTPUTS${RESET}"
echo -e "${YELLOW}Moving results and visualizations to organized folders...${RESET}"

# Move visualization files to visualization directory
mv *.png visualizations/ 2>/dev/null || true
mv parameter_tuning_*.png visualizations/ 2>/dev/null || true
mv championship_odds_comparison.png visualizations/ 2>/dev/null || true
mv seed_distribution_comparison.png visualizations/ 2>/dev/null || true

# Copy report files to results directory
cp backtesting_final_report.json results/ 2>/dev/null || true
cp parameter_tuning_results.json results/ 2>/dev/null || true
cp optimized_parameters.json results/ 2>/dev/null || true
cp optimized_bracket.txt results/ 2>/dev/null || true
cp simulated_bracket.txt results/ 2>/dev/null || true

echo -e "\n${GREEN}WORKFLOW COMPLETE!${RESET}"
echo "Ended at: $(date)"
echo -e "${BLUE}===============================================${RESET}"
echo -e "${YELLOW}Results saved to:${RESET} results/"
echo -e "${YELLOW}Visualizations saved to:${RESET} visualizations/"
echo -e "${YELLOW}Optimized bracket at:${RESET} results/optimized_bracket.txt"
echo -e "${BLUE}===============================================${RESET}"
