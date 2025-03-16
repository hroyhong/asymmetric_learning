"""
Model fitting utilities for the asymmetric learning experiments.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from scipy.optimize import minimize
import sys
import os

# Ensure src is in the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rescorla_wagner import RWModel, RWPlusMinus, ThreeAlphaModel
from config import MODEL_CONFIG
from utils import save_model_params


def negative_log_likelihood(params: np.ndarray, model_class: Any, data: pd.DataFrame, 
                           is_three_alpha: bool = False) -> float:
    """
    Calculate negative log likelihood for model parameters.
    
    Args:
        params: Model parameters to evaluate
        model_class: Class of the model to fit
        data: DataFrame containing experiment data
        is_three_alpha: Whether this is a ThreeAlphaModel
        
    Returns:
        Negative log likelihood
    """
    # Extract parameters based on model type
    if model_class == RWModel:
        learning_rate, initial_value = params
        model = model_class(learning_rate=learning_rate, initial_value=initial_value)
    elif model_class == RWPlusMinus:
        positive_learning_rate, negative_learning_rate, initial_value = params
        model = model_class(positive_learning_rate=positive_learning_rate, 
                           negative_learning_rate=negative_learning_rate,
                           initial_value=initial_value)
    elif model_class == ThreeAlphaModel:
        free_positive_lr, free_negative_lr, forced_lr, initial_value = params
        model = model_class(free_positive_learning_rate=free_positive_lr,
                           free_negative_learning_rate=free_negative_lr,
                           forced_learning_rate=forced_lr,
                           initial_value=initial_value)
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    # Reset model's Q-values
    model.q_values = {}
    
    log_likelihood = 0.0
    
    # Group by different casinos/blocks to reset Q-values between them
    data_groups = data.groupby(['casino']) if 'casino' in data.columns else [('', data)]
    
    for group_name, group_data in data_groups:
        # Reset Q-values for each new casino/block
        model.q_values = {}
        
        # Process each trial in order
        for _, trial in group_data.sort_values('visit').iterrows():
            machine1 = trial['machine1']
            machine2 = trial['machine2']
            chosen_machine = trial['chosen_machine']
            reward = trial['reward']
            
            # Get current Q-values for both machines
            q1 = model.get_q_value(machine1)
            q2 = model.get_q_value(machine2)
            
            # Calculate choice probability using softmax
            p_choose_1 = 1.0 / (1.0 + np.exp(-(q1 - q2)))
            
            # Update log likelihood based on actual choice
            if chosen_machine == machine1:
                log_likelihood += np.log(p_choose_1)
            else:
                log_likelihood += np.log(1.0 - p_choose_1)
            
            # Update model based on choice and reward
            if is_three_alpha and 'is_free_choice' in trial:
                is_free_choice = trial['is_free_choice']
                model.update(chosen_machine, reward, is_free_choice)
            else:
                model.update(chosen_machine, reward)
            
            # For counterfactual learning (Task 2), update unchosen option if known
            if 'counterfactual_reward' in trial and not np.isnan(trial['counterfactual_reward']):
                unchosen_machine = machine2 if chosen_machine == machine1 else machine1
                model.update(unchosen_machine, trial['counterfactual_reward'])
    
    return -log_likelihood


def fit_model(data: pd.DataFrame, model_class: Any, task_name: str, 
              output_dir: str = "results") -> Dict[str, Any]:
    """
    Fit a cognitive model to experimental data.
    
    Args:
        data: DataFrame containing experiment data
        model_class: Class of the model to fit
        task_name: Name of the task
        output_dir: Directory to save results
        
    Returns:
        Dictionary of fitted parameters
    """
    model_type = model_class.__name__.lower()
    model_config = MODEL_CONFIG.get(model_type.replace("model", ""))
    
    if model_class == RWModel:
        # Initialize parameters [learning_rate, initial_value]
        init_params = [0.5, 0.5]
        bounds = [
            model_config["learning_rate_range"],
            model_config["initial_value_range"]
        ]
    elif model_class == RWPlusMinus:
        # Initialize parameters [positive_learning_rate, negative_learning_rate, initial_value]
        init_params = [0.5, 0.5, 0.5]
        bounds = [
            model_config["positive_learning_rate_range"],
            model_config["negative_learning_rate_range"],
            model_config["initial_value_range"]
        ]
    elif model_class == ThreeAlphaModel:
        # Initialize parameters [free_positive_lr, free_negative_lr, forced_lr, initial_value]
        init_params = [0.5, 0.5, 0.5, 0.5]
        bounds = [
            model_config["free_positive_learning_rate_range"],
            model_config["free_negative_learning_rate_range"],
            model_config["forced_learning_rate_range"],
            model_config["initial_value_range"]
        ]
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    is_three_alpha = model_class == ThreeAlphaModel
    
    # Define the objective function
    def objective(params):
        return negative_log_likelihood(params, model_class, data, is_three_alpha)
    
    # Minimize negative log likelihood
    result = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B')
    
    # Extract parameters
    params = result.x
    nll = result.fun
    
    # Calculate BIC
    n_trials = len(data)
    n_params = len(init_params)
    bic = 2 * nll + n_params * np.log(n_trials)
    
    # Create model with optimal parameters
    if model_class == RWModel:
        learning_rate, initial_value = params
        model = model_class(learning_rate=learning_rate, initial_value=initial_value)
        param_dict = {
            "learning_rate": learning_rate,
            "initial_value": initial_value
        }
    elif model_class == RWPlusMinus:
        positive_learning_rate, negative_learning_rate, initial_value = params
        model = model_class(positive_learning_rate=positive_learning_rate, 
                          negative_learning_rate=negative_learning_rate,
                          initial_value=initial_value)
        param_dict = {
            "positive_learning_rate": positive_learning_rate,
            "negative_learning_rate": negative_learning_rate,
            "initial_value": initial_value
        }
    elif model_class == ThreeAlphaModel:
        free_positive_lr, free_negative_lr, forced_lr, initial_value = params
        model = model_class(free_positive_learning_rate=free_positive_lr,
                          free_negative_learning_rate=free_negative_lr,
                          forced_learning_rate=forced_lr,
                          initial_value=initial_value)
        param_dict = {
            "free_positive_learning_rate": free_positive_lr,
            "free_negative_learning_rate": free_negative_lr,
            "forced_learning_rate": forced_lr,
            "initial_value": initial_value
        }
    
    # Add fit metrics
    param_dict.update({
        "negative_log_likelihood": nll,
        "bic": bic,
        "n_trials": n_trials,
        "n_params": n_params
    })
    
    # Save parameters
    save_model_params(param_dict, task_name, model_type, output_dir)
    
    return param_dict


def compare_models(data: pd.DataFrame, model_classes: List[Any], task_name: str, 
                  output_dir: str = "results") -> Dict[str, Any]:
    """
    Fit multiple models and compare them using BIC.
    
    Args:
        data: DataFrame containing experiment data
        model_classes: List of model classes to fit
        task_name: Name of the task
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for model_class in model_classes:
        model_name = model_class.__name__
        print(f"Fitting {model_name}...")
        model_params = fit_model(data, model_class, task_name, output_dir)
        results[model_name] = model_params
    
    # Calculate BIC weights for model comparison
    bic_values = np.array([results[model_name]["bic"] for model_name in results])
    min_bic = np.min(bic_values)
    delta_bic = bic_values - min_bic
    bic_weights = np.exp(-0.5 * delta_bic) / np.sum(np.exp(-0.5 * delta_bic))
    
    comparison_results = {
        "task_name": task_name,
        "n_trials": results[list(results.keys())[0]]["n_trials"],
        "models": []
    }
    
    for i, model_name in enumerate(results):
        comparison_results["models"].append({
            "name": model_name,
            "bic": results[model_name]["bic"],
            "delta_bic": delta_bic[i],
            "bic_weight": bic_weights[i],
            "parameters": results[model_name]
        })
    
    # Save comparison results
    output_path = os.path.join(output_dir, f"{task_name}_model_comparison.json")
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"Model comparison results saved to {output_path}")
    
    return comparison_results 