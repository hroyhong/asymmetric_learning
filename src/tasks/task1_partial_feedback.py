"""
Task 1: Basic 2AFC Task with Partial Feedback

This task replicates the standard 2AFC slot machine paradigm where only the outcome 
of the chosen option is shown. It is used to evaluate whether the model exhibits 
an optimism bias by weighting positive prediction errors more strongly than negative ones.
"""

import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm
import sys
import logging

# Ensure src is in the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import TASK1_CONFIG, SEED
from utils import set_seed, generate_machine_labels, sample_reward, save_results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PartialFeedbackTask:
    """
    Implementation of the Basic Partial Feedback task (Task 1).
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_deepseek: bool = False):
        """
        Initialize the task.
        
        Args:
            config: Configuration parameters
            use_deepseek: Whether to use the DeepSeek model via Volcengine for decision-making
        """
        self.config = config or TASK1_CONFIG
        set_seed(SEED)
        
        self.n_casinos = self.config["n_casinos"]
        self.n_visits_per_casino = self.config["n_visits_per_casino"]
        self.total_visits = self.config["total_visits"]
        self.reward_values = self.config["reward_values"]
        self.reward_probabilities = self.config["reward_probabilities"]
        
        # Assign reward probabilities to casinos
        self.casino_probabilities = {}
        for i, probs in enumerate(self.reward_probabilities):
            casino_id = i + 1
            self.casino_probabilities[casino_id] = probs
        
        # Generate random machine labels for each casino
        self.casino_machines = {}
        for casino_id in range(1, self.n_casinos + 1):
            self.casino_machines[casino_id] = generate_machine_labels(2)
            
        # Set up the model for decision-making
        self.use_deepseek = use_deepseek
        if self.use_deepseek:
            try:
                from api.deepseek import get_api_client
                self.deepseek_client = get_api_client()
                logger.info("Using DeepSeek model for decision-making")
            except ImportError:
                logger.warning("Failed to import DeepSeek client. Falling back to random choice.")
                self.use_deepseek = False
            except Exception as e:
                logger.error(f"Error initializing DeepSeek client: {str(e)}")
                logger.warning("Falling back to random choice.")
                self.use_deepseek = False
    
    def create_prompt(self, history: List[Tuple[str, float]], visit: int, casino_id: int) -> str:
        """
        Create the prompt for the model.
        
        Args:
            history: List of (machine, reward) tuples representing past plays
            visit: Current visit number
            casino_id: Current casino ID
            
        Returns:
            Formatted prompt string
        """
        # Format history
        history_str = ""
        for machine, reward in history:
            history_str += f"- Machine {machine} in Casino {casino_id} delivered {reward:.1f} dollars.\n"
        
        # Get the two machine options for this casino
        machine1, machine2 = self.casino_machines[casino_id]
        
        prompt = f"""You are going to visit four different casinos (named 1, 2, 3, and 4) 24 times each. Each casino owns two slot machines which stochastically return either 0.5 or 0 dollars with different reward probabilities. Your goal is to maximize the sum of received dollars within 96 visits.

You have received the following amount of dollars when playing in the past:
{history_str}

Q: You are now in visit {visit} playing in Casino {casino_id}. Which machine do you choose between Machine {machine1} and Machine {machine2}?
A: Machine """
        
        return prompt
    
    def get_model_choice(self, prompt: str) -> str:
        """
        Get the model's choice from the prompt.
        This uses the DeepSeek model if enabled, otherwise falls back to random choice.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The machine choice (e.g., "A", "B", etc.)
        """
        if self.use_deepseek:
            try:
                # Call the DeepSeek model via Volcengine API
                return self.deepseek_client.get_slot_machine_choice(prompt)
            except Exception as e:
                logger.error(f"Error getting choice from DeepSeek: {str(e)}")
                logger.warning("Falling back to random choice for this trial.")
                # Fall back to random choice if API call fails
        
        # Random choice fallback
        lines = prompt.strip().split('\n')
        question_line = [line for line in lines if line.startswith('Q:')][0]
        machines = question_line.split('between Machine ')[1].split(' and Machine ')
        machine1 = machines[0]
        machine2 = machines[1].rstrip('?')
        
        return random.choice([machine1, machine2])
    
    def run_experiment(self) -> pd.DataFrame:
        """
        Run the experiment and collect data.
        
        Returns:
            DataFrame with trial-by-trial data
        """
        results = []
        visit_counter = 1
        
        for casino_id in tqdm(range(1, self.n_casinos + 1), desc="Casinos"):
            # Get the reward probabilities for this casino
            machine1_prob, machine2_prob = self.casino_probabilities[casino_id]
            machine1_label, machine2_label = self.casino_machines[casino_id]
            probs = {machine1_label: machine1_prob, machine2_label: machine2_prob}
            
            # History of plays for this casino
            history = []
            
            for visit in tqdm(range(1, self.n_visits_per_casino + 1), desc=f"Casino {casino_id} visits", leave=False):
                # Create prompt based on history
                prompt = self.create_prompt(history, visit_counter, casino_id)
                
                # Get model's choice
                chosen_machine = self.get_model_choice(prompt)
                
                # Sample reward based on machine probability
                reward = sample_reward(probs[chosen_machine], self.reward_values)
                
                # Update history
                history.append((chosen_machine, reward))
                
                # Save results
                results.append({
                    'visit': visit_counter,
                    'casino': casino_id,
                    'machine1': machine1_label,
                    'machine2': machine2_label,
                    'machine1_prob': machine1_prob,
                    'machine2_prob': machine2_prob,
                    'chosen_machine': chosen_machine,
                    'reward': reward,
                    'prompt': prompt,
                    'model_used': 'deepseek' if self.use_deepseek else 'random',
                })
                
                visit_counter += 1
        
        return pd.DataFrame(results)
    
    def save_results(self, data: pd.DataFrame, output_dir: str = "results") -> str:
        """
        Save experimental results.
        
        Args:
            data: DataFrame containing experimental results
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        task_name = self.config["name"]
        if self.use_deepseek:
            task_name += "_deepseek"
        return save_results(data, task_name, output_dir)


def main():
    """
    Run Task 1 experiment.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Task 1: Basic Partial Feedback Experiment')
    parser.add_argument('--use-deepseek', action='store_true', help='Use DeepSeek model via Volcengine API')
    args = parser.parse_args()
    
    print("Running Task 1: Basic Partial Feedback Experiment")
    print(f"Using DeepSeek model: {args.use_deepseek}")
    
    task = PartialFeedbackTask(use_deepseek=args.use_deepseek)
    results = task.run_experiment()
    filepath = task.save_results(results)
    print(f"Experiment completed. Results saved to {filepath}")
    

if __name__ == "__main__":
    main() 