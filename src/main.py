"""
Main script to run the asymmetric learning experiments.
"""

import os
import sys
import argparse
from config import SEED
from utils import set_seed


def run_task1():
    """Run Task 1: Basic Partial Feedback"""
    from tasks.task1_partial_feedback import PartialFeedbackTask
    from models.rescorla_wagner import RWModel, RWPlusMinus
    from models.model_fitting import compare_models
    
    # Run experiment
    task = PartialFeedbackTask()
    print("Running Task 1: Basic Partial Feedback Experiment")
    results = task.run_experiment()
    filepath = task.save_results(results)
    print(f"Experiment completed. Results saved to {filepath}")
    
    # Fit models
    print("\nFitting cognitive models to Task 1 data...")
    model_comparison = compare_models(
        data=results,
        model_classes=[RWModel, RWPlusMinus],
        task_name=task.config["name"]
    )
    
    # Check for optimism bias
    rw_plus_minus_results = None
    for model_info in model_comparison["models"]:
        if model_info["name"] == "RWPlusMinus":
            rw_plus_minus_results = model_info
    
    if rw_plus_minus_results:
        positive_lr = rw_plus_minus_results["parameters"]["positive_learning_rate"]
        negative_lr = rw_plus_minus_results["parameters"]["negative_learning_rate"]
        
        print("\nTask 1 Results Summary:")
        print(f"Positive learning rate (α⁺): {positive_lr:.4f}")
        print(f"Negative learning rate (α⁻): {negative_lr:.4f}")
        
        if positive_lr > negative_lr:
            print("Optimism bias detected: Model shows stronger learning from positive outcomes.")
        elif positive_lr < negative_lr:
            print("Pessimism bias detected: Model shows stronger learning from negative outcomes.")
        else:
            print("No asymmetric learning detected: Equal learning from positive and negative outcomes.")


def run_task2():
    """Run Task 2: Full Feedback with counterfactual information"""
    print("Task 2: Full Feedback implementation will be added.")
    # To be implemented


def run_task3():
    """Run Task 3: Agency Condition"""
    print("Task 3: Agency Condition implementation will be added.")
    # To be implemented


def main():
    """Main function to run experiments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run asymmetric learning experiments")
    parser.add_argument(
        "--task", 
        type=int, 
        choices=[1, 2, 3, 0], 
        default=0,
        help="Task to run (1=Partial Feedback, 2=Full Feedback, 3=Agency Condition, 0=All Tasks)"
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(SEED)
    
    # Run selected task(s)
    if args.task == 0 or args.task == 1:
        run_task1()
        print("\n" + "="*50 + "\n")
        
    if args.task == 0 or args.task == 2:
        run_task2()
        print("\n" + "="*50 + "\n")
        
    if args.task == 0 or args.task == 3:
        run_task3()


if __name__ == "__main__":
    main() 