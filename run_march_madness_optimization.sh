# run_march_madness_optimization.sh
#!/bin/bash

set -e  # Exit immediately on error

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

# Ensure directories exist
echo -e "${YELLOW}Creating necessary directories...${RESET}"
mkdir -p historical_data results visualizations

# Step 1: Data Collection
echo -e "\n${GREEN}STEP 1: COLLECTING CURRENT SEASON DATA${RESET}"
echo -e "${YELLOW}Fetching team info, statistics, and rankings from ESPN...${RESET}"
python3 data_collection.py

# Step 2: Run Backtesting with Real Data
echo -e "\n${GREEN}STEP 2: RUNNING BACKTESTING FOR PARAMETER OPTIMIZATION${RESET}"
echo -e "${YELLOW}Testing against historical tournaments (2015-2023)...${RESET}"
python3 run_backtesting.py --years 2015,2016,2017,2018,2019,2020,2021,2022,2023 --sims 500 --datadir historical_data

# Step 3: Extract optimized parameters and save to a file
echo -e "\n${GREEN}STEP 3: EXTRACTING OPTIMIZED PARAMETERS${RESET}"
echo -e "${YELLOW}Reading optimal parameters from backtesting report...${RESET}"

if command -v jq &> /dev/null; then
    UPSET_FACTOR=$(jq -r '.optimized_parameters.UPSET_FACTOR' backtesting_final_report.json)
    MOMENTUM_FACTOR=$(jq -r '.optimized_parameters.MOMENTUM_FACTOR' backtesting_final_report.json)
    
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
fi

# Step 4: Run full tournament simulation with optimized parameters
echo -e "\n${GREEN}STEP 4: RUNNING FULL TOURNAMENT SIMULATION WITH OPTIMIZED PARAMETERS${RESET}"
echo -e "${YELLOW}Simulating 10,000 tournament runs with optimized parameters...${RESET}"
python3 run_final_simulation.py --sims 10000 --params optimized_parameters.json

# Step 5: Save and Validate Results
echo -e "\n${GREEN}STEP 5: VALIDATING FINAL BRACKET RESULTS${RESET}"
echo -e "${YELLOW}Ensuring only real teams appear in the final bracket...${RESET}"
python3 validate_bracket.py

echo -e "\n${GREEN}ALL STEPS COMPLETED SUCCESSFULLY${RESET}"
