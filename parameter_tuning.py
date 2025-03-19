import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ParameterTuner:
    """
    A utility for optimizing parameters in the March Madness simulator
    based on backtesting results.
    """
    
    def __init__(self, backtester):
        """
        Initialize with a backtester.
        
        Args:
            backtester: MarchMadnessBacktester instance
        """
        self.backtester = backtester
        self.tuning_results = {}
    
    def tune_upset_factor(self, values=None, simulations=200):
        """
        Find the optimal UPSET_FACTOR value.
        
        Args:
            values: List of values to test (default: range from 0.05 to 0.40)
            simulations: Number of simulations for each test
            
        Returns:
            DataFrame with results and optimal value
        """
        print("Tuning UPSET_FACTOR parameter...")
        
        if values is None:
            values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        
        results = self.backtester.backtest_parameter('UPSET_FACTOR', values)
        self.tuning_results['UPSET_FACTOR'] = results
        
        # Find the best value
        best_idx = results['bracket_score'].idxmax()
        best_value = results.loc[best_idx, 'parameter_value']
        best_score = results.loc[best_idx, 'bracket_score']
        
        print(f"Optimal UPSET_FACTOR: {best_value} (Score: {best_score:.2f})")
        
        # Plot the results
        self._plot_parameter_results(results, 'UPSET_FACTOR', 
                                     'Impact of Upset Factor on Bracket Performance')
        
        return results, best_value
    
    def tune_momentum_factor(self, values=None, simulations=200):
        """
        Find the optimal MOMENTUM_FACTOR value.
        
        Args:
            values: List of values to test (default: range from 0.0 to 0.20)
            simulations: Number of simulations for each test
            
        Returns:
            DataFrame with results and optimal value
        """
        print("Tuning MOMENTUM_FACTOR parameter...")
        
        if values is None:
            values = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        
        results = self.backtester.backtest_parameter('MOMENTUM_FACTOR', values)
        self.tuning_results['MOMENTUM_FACTOR'] = results
        
        # Find the best value
        best_idx = results['bracket_score'].idxmax()
        best_value = results.loc[best_idx, 'parameter_value']
        best_score = results.loc[best_idx, 'bracket_score']
        
        print(f"Optimal MOMENTUM_FACTOR: {best_value} (Score: {best_score:.2f})")
        
        # Plot the results
        self._plot_parameter_results(results, 'MOMENTUM_FACTOR', 
                                    'Impact of Momentum Factor on Bracket Performance')
        
        return results, best_value
    
    def tune_round_variance(self, round_num, values=None, simulations=200):
        """
        Find the optimal round-specific variance factor.
        
        Args:
            round_num: Tournament round (1-6)
            values: List of values to test
            simulations: Number of simulations for each test
            
        Returns:
            DataFrame with results and optimal value
        """
        param_name = f"variance_factors.{round_num}"
        round_names = {
            1: "First Round",
            2: "Second Round",
            3: "Sweet 16",
            4: "Elite 8",
            5: "Final Four",
            6: "Championship"
        }
        
        print(f"Tuning variance factor for {round_names.get(round_num, f'Round {round_num}')}...")
        
        if values is None:
            # Default range depends on the round
            if round_num <= 2:
                values = [0.8, 1.0, 1.2, 1.4, 1.6]
            elif round_num <= 4:
                values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            else:
                values = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        results = self.backtester.backtest_parameter(param_name, values)
        self.tuning_results[param_name] = results
        
        # Find the best value
        best_idx = results['bracket_score'].idxmax()
        best_value = results.loc[best_idx, 'parameter_value']
        best_score = results.loc[best_idx, 'bracket_score']
        
        print(f"Optimal variance factor for {round_names.get(round_num, f'Round {round_num}')}: "
              f"{best_value} (Score: {best_score:.2f})")
        
        # Plot the results
        self._plot_parameter_results(results, f"Variance Factor (Round {round_num})", 
                                    f'Impact of Round {round_num} Variance on Bracket Performance')
        
        return results, best_value
    
    def tune_seed_performance_adjustment(self, seed, values=None, simulations=200):
        """
        Find the optimal seed-specific performance adjustment factor.
        
        Args:
            seed: Seed number (1-16)
            values: List of values to test
            simulations: Number of simulations for each test
            
        Returns:
            DataFrame with results and optimal value
        """
        param_name = f"seed_performance_adjustments.{seed}"
        
        print(f"Tuning performance adjustment for Seed {seed}...")
        
        if values is None:
            # Default range depends on the seed
            if seed <= 4:
                values = [0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
            elif seed <= 8:
                values = [0.9, 0.95, 1.0, 1.05, 1.1]
            elif seed <= 12:
                values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            else:
                values = [0.7, 0.8, 0.9, 1.0, 1.1]
        
        results = self.backtester.backtest_parameter(param_name, values)
        self.tuning_results[param_name] = results
        
        # Find the best value
        best_idx = results['bracket_score'].idxmax()
        best_value = results.loc[best_idx, 'parameter_value']
        best_score = results.loc[best_idx, 'bracket_score']
        
        print(f"Optimal performance adjustment for Seed {seed}: "
              f"{best_value} (Score: {best_score:.2f})")
        
        # Plot the results
        self._plot_parameter_results(results, f"Seed {seed} Adjustment", 
                                    f'Impact of Seed {seed} Adjustment on Bracket Performance')
        
        return results, best_value
    
    def tune_historical_matchup_weight(self, values=None, simulations=200):
        """
        Find the optimal weight for historical matchup data.
        
        Args:
            values: List of values to test
            simulations: Number of simulations for each test
            
        Returns:
            DataFrame with results and optimal value
        """
        print("Tuning historical matchup weight parameter...")
        
        if values is None:
            values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        results = self.backtester.backtest_parameter('historical_matchup_weights', values)
        self.tuning_results['historical_matchup_weights'] = results
        
        # Find the best value
        best_idx = results['bracket_score'].idxmax()
        best_value = results.loc[best_idx, 'parameter_value']
        best_score = results.loc[best_idx, 'bracket_score']
        
        print(f"Optimal historical matchup weight: {best_value} (Score: {best_score:.2f})")
        
        # Plot the results
        self._plot_parameter_results(results, 'Historical Matchup Weight', 
                                    'Impact of Historical Data Weight on Bracket Performance')
        
        return results, best_value
    
    def run_optimization(self):
        """
        Run a comprehensive optimization of all major parameters.
        
        Returns:
            Dictionary with optimal parameter values
        """
        print("Starting comprehensive parameter optimization...")
        
        # Optimize major parameters
        _, best_upset = self.tune_upset_factor()
        _, best_momentum = self.tune_momentum_factor()
        
        # Optimize round-specific variance factors
        round_variance = {}
        for round_num in range(1, 7):
            _, best_var = self.tune_round_variance(round_num)
            round_variance[round_num] = best_var
        
        # Optimize key seed adjustments
        seed_adjustments = {}
        for seed in [1, 5, 8, 12, 16]:  # Most important seeds
            _, best_adj = self.tune_seed_performance_adjustment(seed)
            seed_adjustments[seed] = best_adj
        
        # Optimize historical matchup weight
        _, best_hist_weight = self.tune_historical_matchup_weight()
        
        # Compile optimal parameters
        optimal_params = {
            'UPSET_FACTOR': best_upset,
            'MOMENTUM_FACTOR': best_momentum,
            'variance_factors': round_variance,
            'seed_performance_adjustments': seed_adjustments,
            'historical_matchup_weights': best_hist_weight
        }
        
        print("\nOptimization complete!")
        print("Optimal parameter values:")
        for param, value in optimal_params.items():
            print(f"  {param}: {value}")
        
        return optimal_params
    
    def run_grid_search(self, param_grid=None):
        """
        Run a grid search to find the best parameter combination.
        
        Args:
            param_grid: Dictionary mapping parameters to lists of values to test
                        (None for default grid)
        
        Returns:
            Dictionary with grid search results
        """
        print("Running grid search over multiple parameters...")
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'UPSET_FACTOR': [0.15, 0.20, 0.25],
                'MOMENTUM_FACTOR': [0.05, 0.08, 0.12],
                'variance_factors.3': [1.2, 1.4, 1.6],  # Sweet 16 variance
                'variance_factors.5': [1.5, 1.8, 2.1]   # Final Four variance
            }
        
        # Run grid search
        grid_results = self.backtester.grid_search(param_grid)
        
        # Store the results
        self.tuning_results['grid_search'] = grid_results
        
        print("\nGrid search complete!")
        print("Best parameter combination:")
        for param, value in grid_results['best_parameters'].items():
            print(f"  {param}: {value}")
        print(f"Best bracket score: {grid_results['best_score']:.2f}")
        
        return grid_results
    
    def _plot_parameter_results(self, results_df, param_name, title):
        """
        Create a plot showing parameter impact on performance.
        
        Args:
            results_df: DataFrame with results
            param_name: Name of the parameter
            title: Plot title
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Set up twin axes for different scales
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Plot bracket score on first axis
            sns.lineplot(x='parameter_value', y='bracket_score', data=results_df, 
                        marker='o', color='blue', ax=ax1)
            ax1.set_ylabel('Bracket Score', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Plot accuracy metrics on second axis
            sns.lineplot(x='parameter_value', y='accuracy', data=results_df, 
                        marker='s', color='green', ax=ax2, label='Overall Accuracy')
            sns.lineplot(x='parameter_value', y='final_four_accuracy', data=results_df, 
                        marker='^', color='red', ax=ax2, label='Final Four Accuracy')
            ax2.set_ylabel('Accuracy (%)', color='black')
            
            # Add best parameter marker
            best_idx = results_df['bracket_score'].idxmax()
            best_value = results_df.loc[best_idx, 'parameter_value']
            best_score = results_df.loc[best_idx, 'bracket_score']
            
            ax1.axvline(x=best_value, color='gray', linestyle='--', alpha=0.7)
            ax1.plot(best_value, best_score, 'o', color='orange', markersize=10)
            ax1.annotate(f'Best: {best_value}', 
                        xy=(best_value, best_score),
                        xytext=(5, 10),
                        textcoords='offset points',
                        fontsize=10,
                        backgroundcolor='w')
            
            # Formatting
            plt.title(title)
            ax1.set_xlabel(param_name)
            ax1.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"parameter_tuning_{param_name.replace('.', '_')}.png", dpi=300)
            print(f"Saved plot to parameter_tuning_{param_name.replace('.', '_')}.png")
            
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def analyze_interactions(self, param1, param2, metric='bracket_score'):
        """
        Analyze interactions between two parameters.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            metric: Performance metric to analyze
            
        Returns:
            DataFrame with interaction results
        """
        print(f"Analyzing interaction between {param1} and {param2}...")
        
        # Check if we have grid search results that include these parameters
        if 'grid_search' not in self.tuning_results:
            print("No grid search results available. Run grid_search first.")
            return None
        
        # Extract results
        grid_results = self.tuning_results['grid_search']['all_results']
        
        # Filter for the relevant parameters
        filtered_results = []
        for result in grid_results:
            if param1 in result['parameters'] and param2 in result['parameters']:
                filtered_results.append({
                    param1: result['parameters'][param1],
                    param2: result['parameters'][param2],
                    metric: result[metric]
                })
        
        if not filtered_results:
            print(f"No results found containing both {param1} and {param2}")
            return None
        
        # Convert to DataFrame
        interaction_df = pd.DataFrame(filtered_results)
        
        # Create heatmap
        try:
            # Pivot the data for the heatmap
            pivot_df = interaction_df.pivot_table(index=param1, columns=param2, values=metric)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
            plt.title(f'Interaction between {param1} and {param2} on {metric}')
            plt.tight_layout()
            
            # Save the plot
            interaction_file = f"interaction_{param1}_{param2}.png"
            plt.savefig(interaction_file, dpi=300)
            print(f"Saved interaction plot to {interaction_file}")
            
        except Exception as e:
            print(f"Error creating interaction plot: {e}")
        
        return interaction_df
    
    def get_recommended_parameters(self):
        """
        Get the recommended parameter settings based on all tuning results.
        
        Returns:
            Dictionary with recommended parameter values
        """
        if not self.tuning_results:
            print("No tuning results available. Run optimization first.")
            return None
        
        # If we have grid search results, use the best combination
        if 'grid_search' in self.tuning_results:
            return self.tuning_results['grid_search']['best_parameters']
        
        # Otherwise, compile best values from individual parameter tuning
        recommended = {}
        
        # Extract best values for each parameter
        for param, results in self.tuning_results.items():
            if param != 'grid_search':
                try:
                    best_idx = results['bracket_score'].idxmax()
                    recommended[param] = results.loc[best_idx, 'parameter_value']
                except:
                    pass
        
        # Handle nested parameters (variance_factors, seed_adjustments)
        variance_factors = {}
        seed_adjustments = {}
        historical_weight = None
        
        for param, value in recommended.items():
            if param.startswith('variance_factors.'):
                round_num = int(param.split('.')[1])
                variance_factors[round_num] = value
            elif param.startswith('seed_performance_adjustments.'):
                seed = int(param.split('.')[1])
                seed_adjustments[seed] = value
            elif param == 'historical_matchup_weights':
                historical_weight = value
        
        # Compile the final recommended parameters
        final_recommended = {}
        
        if 'UPSET_FACTOR' in recommended:
            final_recommended['UPSET_FACTOR'] = recommended['UPSET_FACTOR']
        
        if 'MOMENTUM_FACTOR' in recommended:
            final_recommended['MOMENTUM_FACTOR'] = recommended['MOMENTUM_FACTOR']
        
        if variance_factors:
            final_recommended['variance_factors'] = variance_factors
        
        if seed_adjustments:
            final_recommended['seed_performance_adjustments'] = seed_adjustments
        
        if historical_weight is not None:
            final_recommended['historical_matchup_weights'] = historical_weight
        
        print("\nRecommended parameter values:")
        for param, value in final_recommended.items():
            print(f"  {param}: {value}")
        
        return final_recommended
    
    def save_tuning_results(self, filename="parameter_tuning_results.json"):
        """
        Save tuning results to a JSON file.
        
        Args:
            filename: Path to save the file
        """
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        
        for param, results in self.tuning_results.items():
            if param == 'grid_search':
                serializable_results[param] = {
                    'best_parameters': results['best_parameters'],
                    'best_score': results['best_score']
                }
            else:
                serializable_results[param] = results.to_dict('records')
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved tuning results to {filename}")
    
    def load_tuning_results(self, filename="parameter_tuning_results.json"):
        """
        Load tuning results from a JSON file.
        
        Args:
            filename: Path to the file
        """
        import json
        import os
        
        if not os.path.exists(filename):
            print(f"No tuning results file found at {filename}")
            return
        
        with open(filename, 'r') as f:
            serialized_results = json.load(f)
        
        # Convert back to appropriate format
        for param, results in serialized_results.items():
            if param == 'grid_search':
                self.tuning_results[param] = results
            else:
                self.tuning_results[param] = pd.DataFrame(results)
        
        print(f"Loaded tuning results from {filename}")
    
    def apply_parameters_to_simulator(self, simulator, parameters=None):
        """
        Apply tuned parameters to a simulator instance.
        
        Args:
            simulator: The simulator object to update
            parameters: Parameter dictionary (None to use recommended parameters)
            
        Returns:
            Updated simulator object
        """
        if parameters is None:
            parameters = self.get_recommended_parameters()
            
        if parameters is None:
            print("No parameters available to apply")
            return simulator
        
        print("Applying optimized parameters to simulator...")
        
        # Apply direct parameters
        if 'UPSET_FACTOR' in parameters and hasattr(simulator, 'UPSET_FACTOR'):
            simulator.UPSET_FACTOR = parameters['UPSET_FACTOR']
            print(f"  Set UPSET_FACTOR to {parameters['UPSET_FACTOR']}")
        
        if 'MOMENTUM_FACTOR' in parameters and hasattr(simulator, 'MOMENTUM_FACTOR'):
            simulator.MOMENTUM_FACTOR = parameters['MOMENTUM_FACTOR']
            print(f"  Set MOMENTUM_FACTOR to {parameters['MOMENTUM_FACTOR']}")
        
        # Apply variance factors
        if 'variance_factors' in parameters and hasattr(simulator, 'apply_round_variance'):
            # May need custom handling depending on implementation
            print(f"  Set variance factors: {parameters['variance_factors']}")
            # Example: simulator.variance_factors = parameters['variance_factors']
        
        # Apply seed adjustments
        if 'seed_performance_adjustments' in parameters and hasattr(simulator, 'apply_seed_adjustment'):
            # May need custom handling depending on implementation
            print(f"  Set seed adjustments: {parameters['seed_performance_adjustments']}")
            # Example: simulator.seed_performance_factors = parameters['seed_performance_adjustments']
        
        # Apply historical matchup weight
        if 'historical_matchup_weights' in parameters and hasattr(simulator, 'apply_historical_matchup'):
            # May need custom handling depending on implementation
            print(f"  Set historical matchup weight: {parameters['historical_matchup_weights']}")
            # Example: simulator.historical_weight = parameters['historical_matchup_weights']
        
        return simulator
    
    def compare_to_baseline(self, baseline_params, optimized_params=None):
        """
        Compare optimized parameters to baseline parameters.
        
        Args:
            baseline_params: Dictionary of baseline parameters
            optimized_params: Dictionary of optimized parameters (None to use recommended)
            
        Returns:
            DataFrame with comparison results
        """
        if optimized_params is None:
            optimized_params = self.get_recommended_parameters()
            
        if optimized_params is None:
            print("No optimized parameters available")
            return None
        
        print("Comparing optimized parameters to baseline...")
        
        # Save current parameters
        original_params = self.backtester.parameters.copy()
        
        # Test baseline parameters
        self.backtester.parameters = baseline_params
        baseline_metrics = self.backtester.run_backtests(num_simulations=200)
        
        # Test optimized parameters
        self.backtester.parameters = optimized_params
        optimized_metrics = self.backtester.run_backtests(num_simulations=200)
        
        # Reset to original parameters
        self.backtester.parameters = original_params
        
        # Compile comparison
        comparison = pd.DataFrame({
            'Metric': ['Bracket Score', 'Prediction Accuracy', 'Final Four Accuracy', 
                     'Champion Accuracy', 'Upset Detection Rate'],
            'Baseline': [baseline_metrics['average_bracket_score'],
                        baseline_metrics['average_accuracy'],
                        baseline_metrics['final_four_accuracy'],
                        baseline_metrics['champion_accuracy'],
                        baseline_metrics['upset_detection_rate']],
            'Optimized': [optimized_metrics['average_bracket_score'],
                         optimized_metrics['average_accuracy'],
                         optimized_metrics['final_four_accuracy'],
                         optimized_metrics['champion_accuracy'],
                         optimized_metrics['upset_detection_rate']]
        })
        
        # Calculate improvement
        comparison['Improvement'] = comparison['Optimized'] - comparison['Baseline']
        comparison['Improvement %'] = (comparison['Improvement'] / comparison['Baseline'] * 100).round(1)
        
        print("\nPerformance comparison:")
        print(comparison)
        
        # Plot comparison
        try:
            plt.figure(figsize=(12, 6))
            
            metrics = comparison['Metric']
            baseline = comparison['Baseline']
            optimized = comparison['Optimized']
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            rects1 = ax.bar(x - width/2, baseline, width, label='Baseline')
            rects2 = ax.bar(x + width/2, optimized, width, label='Optimized')
            
            ax.set_title('Performance Comparison: Baseline vs. Optimized Parameters')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            # Add value labels
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(rect.get_x() + rect.get_width()/2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            fig.tight_layout()
            
            # Save the plot
            plt.savefig("parameter_comparison.png", dpi=300)
            print("Saved comparison plot to parameter_comparison.png")
            
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
        
        return comparison
