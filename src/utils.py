"""
Utility functions for the asymmetric learning project.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
import json
from datetime import datetime


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_machine_labels(n_machines: int = 2) -> List[str]:
    """
    Generate random labels for slot machines.
    
    Args:
        n_machines: Number of machines to label
        
    Returns:
        List of machine labels
    """
    # Use uppercase letters from the alphabet
    available_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    random.shuffle(available_labels)
    return available_labels[:n_machines]


def sample_reward(probability: float, reward_values: List[float]) -> float:
    """
    Sample a reward based on given probability.
    
    Args:
        probability: Probability of getting the higher reward
        reward_values: List of possible reward values [high_reward, low_reward]
        
    Returns:
        Sampled reward value
    """
    if random.random() < probability:
        return reward_values[0]  # Higher reward
    else:
        return reward_values[1]  # Lower reward


def save_results(data: pd.DataFrame, task_name: str, output_dir: str = "results") -> str:
    """
    Save experimental results to CSV file.
    
    Args:
        data: DataFrame containing experimental results
        task_name: Name of the task
        output_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    data.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return filepath


def save_model_params(model_params: Dict[str, Any], task_name: str, 
                      model_name: str, output_dir: str = "results") -> str:
    """
    Save model parameters to JSON file.
    
    Args:
        model_params: Dictionary of model parameters
        task_name: Name of the task
        model_name: Name of the model
        output_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_name}_{model_name}_params_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(model_params, f, indent=4)
    
    print(f"Model parameters saved to {filepath}")
    return filepath


def create_prompt_history(history: List[Tuple[str, str, float, Optional[float]]]) -> str:
    """
    Create a prompt history string from the history list.
    
    Args:
        history: List of (machine, choice_type, reward, counterfactual_reward) tuples
                choice_type can be "you played" or "someone else played"
    
    Returns:
        Formatted history string
    """
    history_lines = []
    
    for i, (machine, choice_type, reward, counterfactual_reward) in enumerate(history):
        visit_num = i + 1
        line = f"- On visit {visit_num}, {choice_type} Machine {machine} and earned {reward:.1f} point"
        
        if counterfactual_reward is not None:
            # This is for full feedback condition
            counterfactual_machine = "Unknown"  # This should be updated in the actual implementation
            line += f". On Machine {counterfactual_machine}, you would have earned {counterfactual_reward:.1f} point."
        else:
            line += "."
            
        history_lines.append(line)
    
    return "\n".join(history_lines) 