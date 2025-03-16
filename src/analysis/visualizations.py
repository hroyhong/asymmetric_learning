"""
Visualization utilities for analyzing asymmetric learning experiment results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
import sys

# Ensure src is in the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ANALYSIS_CONFIG, OUTPUT_DIR


def plot_learning_rates(model_params: Dict[str, Any], title: str = "Learning Rates Comparison",
                       save_path: Optional[str] = None) -> None:
    """
    Plot comparison of learning rates.
    
    Args:
        model_params: Dictionary of model parameters
        title: Plot title
        save_path: Path to save the figure (if None, display only)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract learning rates based on model type
    if "positive_learning_rate" in model_params and "negative_learning_rate" in model_params:
        # RWPlusMinus model
        labels = ["Positive (α⁺)", "Negative (α⁻)"]
        values = [model_params["positive_learning_rate"], model_params["negative_learning_rate"]]
        colors = ["green", "red"]
    elif "free_positive_learning_rate" in model_params:
        # ThreeAlphaModel
        labels = ["Free Positive (α⁺)", "Free Negative (α⁻)", "Forced (α)"]
        values = [
            model_params["free_positive_learning_rate"],
            model_params["free_negative_learning_rate"],
            model_params["forced_learning_rate"]
        ]
        colors = ["green", "red", "blue"]
    else:
        # RWModel
        labels = ["Learning Rate (α)"]
        values = [model_params["learning_rate"]]
        colors = ["blue"]
    
    # Create bar plot
    plt.bar(labels, values, color=colors, alpha=0.7)
    plt.ylim(0, 1)
    plt.ylabel("Learning Rate")
    plt.title(title)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f"{v:.3f}", ha='center')
    
    # Add horizontal line at 0.5 for reference
    plt.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_learning_curves(data: pd.DataFrame, window_size: int = 5,
                        title: str = "Learning Curves", 
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curves showing performance over trials.
    
    Args:
        data: DataFrame with experimental results
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save the figure (if None, display only)
    """
    plt.figure(figsize=(12, 6))
    
    # Compute whether the chosen option was optimal (had higher reward probability)
    data['optimal_choice'] = False
    
    # Get list of unique casinos or blocks
    if 'casino' in data.columns:
        groups = data['casino'].unique()
        group_col = 'casino'
    else:
        groups = [1]  # Default if no casino/block grouping
        group_col = None
    
    for group in groups:
        if group_col:
            group_data = data[data[group_col] == group]
        else:
            group_data = data
            
        # Determine optimal choice for each trial
        for i, row in group_data.iterrows():
            machine1_prob = row['machine1_prob']
            machine2_prob = row['machine2_prob']
            chosen_machine = row['chosen_machine']
            
            if (machine1_prob > machine2_prob and chosen_machine == row['machine1']) or \
               (machine2_prob > machine1_prob and chosen_machine == row['machine2']):
                data.at[i, 'optimal_choice'] = True
    
    # Calculate moving average of optimal choices
    data['optimal_choice_int'] = data['optimal_choice'].astype(int)
    data['optimal_moving_avg'] = data['optimal_choice_int'].rolling(window=window_size, min_periods=1).mean()
    
    # Plot by casino/block if applicable
    if group_col:
        for group in groups:
            group_data = data[data[group_col] == group]
            plt.plot(group_data['visit'], group_data['optimal_moving_avg'], 
                    label=f"{group_col.capitalize()} {group}")
    else:
        plt.plot(data['visit'], data['optimal_moving_avg'])
    
    plt.xlabel("Trial")
    plt.ylabel(f"Proportion of Optimal Choices (Moving Avg, window={window_size})")
    plt.title(title)
    
    if group_col:
        plt.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_model_comparison(comparison_results: Dict[str, Any], 
                         title: str = "Model Comparison",
                         save_path: Optional[str] = None) -> None:
    """
    Plot model comparison using BIC weights.
    
    Args:
        comparison_results: Dictionary with model comparison results
        title: Plot title
        save_path: Path to save the figure (if None, display only)
    """
    plt.figure(figsize=(10, 6))
    
    models = [model["name"] for model in comparison_results["models"]]
    bic_weights = [model["bic_weight"] for model in comparison_results["models"]]
    
    # Sort by BIC weight
    sorted_indices = np.argsort(bic_weights)[::-1]  # Descending order
    models = [models[i] for i in sorted_indices]
    bic_weights = [bic_weights[i] for i in sorted_indices]
    
    # Create bar plot
    bars = plt.bar(models, bic_weights, color='skyblue', alpha=0.7)
    
    # Add value labels
    for bar, weight in zip(bars, bic_weights):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f"{weight:.3f}", ha='center')
    
    plt.ylim(0, 1.1)
    plt.ylabel("BIC Weight")
    plt.title(title)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def analyze_task_results(results_file: str, model_comparison_file: Optional[str] = None,
                        output_dir: str = OUTPUT_DIR) -> None:
    """
    Analyze and visualize results from a task.
    
    Args:
        results_file: Path to the CSV file with experiment results
        model_comparison_file: Path to the JSON file with model comparison results
        output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load results data
    data = pd.read_csv(results_file)
    task_name = os.path.basename(results_file).split('_')[0]
    
    # Plot learning curves
    learning_curve_title = f"{task_name.replace('_', ' ').title()} Learning Curves"
    learning_curve_path = os.path.join(output_dir, f"{task_name}_learning_curves.png")
    plot_learning_curves(data, title=learning_curve_title, save_path=learning_curve_path)
    
    # Plot model comparison if available
    if model_comparison_file and os.path.exists(model_comparison_file):
        with open(model_comparison_file, 'r') as f:
            comparison_results = json.load(f)
        
        model_comparison_title = f"{task_name.replace('_', ' ').title()} Model Comparison"
        model_comparison_path = os.path.join(output_dir, f"{task_name}_model_comparison.png")
        plot_model_comparison(comparison_results, title=model_comparison_title, 
                             save_path=model_comparison_path)
        
        # Plot learning rates for the best model
        best_model = max(comparison_results["models"], key=lambda x: x["bic_weight"])
        learning_rates_title = f"Learning Rates for {best_model['name']}"
        learning_rates_path = os.path.join(output_dir, f"{task_name}_{best_model['name']}_learning_rates.png")
        plot_learning_rates(best_model["parameters"], title=learning_rates_title, 
                          save_path=learning_rates_path)


def main(task_results_dir: str = "results"):
    """
    Run analysis on all task results in the specified directory.
    
    Args:
        task_results_dir: Directory containing task results
    """
    # Find all CSV result files
    result_files = [f for f in os.listdir(task_results_dir) if f.endswith('.csv')]
    
    for result_file in result_files:
        task_name = result_file.split('_')[0]
        
        # Look for corresponding model comparison file
        model_comparison_file = os.path.join(task_results_dir, f"{task_name}_model_comparison.json")
        model_comparison_path = model_comparison_file if os.path.exists(model_comparison_file) else None
        
        print(f"Analyzing results for {task_name}...")
        analyze_task_results(
            os.path.join(task_results_dir, result_file),
            model_comparison_path,
            task_results_dir
        )
        print(f"Analysis for {task_name} completed.")


if __name__ == "__main__":
    main() 