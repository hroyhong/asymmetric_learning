"""
Implementation of Rescorla-Wagner (RW) reinforcement learning models.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union


class RWModel:
    """
    Standard Rescorla-Wagner reinforcement learning model with a single learning rate.
    """
    
    def __init__(self, learning_rate: float = 0.1, initial_value: float = 0.5):
        """
        Initialize the RW model.
        
        Args:
            learning_rate: Learning rate (alpha) for all prediction errors
            initial_value: Initial Q-value for all machines
        """
        self.learning_rate = learning_rate
        self.initial_value = initial_value
        self.q_values = {}  # Dictionary to store Q-values for each machine
    
    def initialize_machine(self, machine_id: str) -> None:
        """
        Initialize Q-value for a new machine.
        
        Args:
            machine_id: Identifier for the machine
        """
        if machine_id not in self.q_values:
            self.q_values[machine_id] = self.initial_value
    
    def get_q_value(self, machine_id: str) -> float:
        """
        Get the Q-value for a specific machine.
        
        Args:
            machine_id: Identifier for the machine
            
        Returns:
            Current Q-value
        """
        self.initialize_machine(machine_id)
        return self.q_values[machine_id]
    
    def update(self, machine_id: str, reward: float) -> float:
        """
        Update the Q-value based on received reward.
        
        Args:
            machine_id: Identifier for the machine
            reward: Received reward
            
        Returns:
            Prediction error
        """
        self.initialize_machine(machine_id)
        current_value = self.q_values[machine_id]
        prediction_error = reward - current_value
        self.q_values[machine_id] += self.learning_rate * prediction_error
        return prediction_error
    
    def choose(self, machine_ids: List[str]) -> str:
        """
        Choose a machine based on Q-values (greedy policy).
        
        Args:
            machine_ids: List of available machine identifiers
            
        Returns:
            Chosen machine identifier
        """
        # Initialize any new machines
        for machine_id in machine_ids:
            self.initialize_machine(machine_id)
        
        # Choose the machine with the highest Q-value
        q_values = [self.q_values[machine_id] for machine_id in machine_ids]
        max_value = max(q_values)
        # If multiple machines have the same max value, choose randomly among them
        best_machines = [machine_id for i, machine_id in enumerate(machine_ids) if q_values[i] == max_value]
        return np.random.choice(best_machines)
    
    def get_params(self) -> Dict[str, float]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameter values
        """
        return {
            "learning_rate": self.learning_rate,
            "initial_value": self.initial_value
        }


class RWPlusMinus(RWModel):
    """
    Asymmetric Rescorla-Wagner model with separate learning rates for positive and negative prediction errors.
    """
    
    def __init__(self, 
                 positive_learning_rate: float = 0.1, 
                 negative_learning_rate: float = 0.1, 
                 initial_value: float = 0.5):
        """
        Initialize the RW± model.
        
        Args:
            positive_learning_rate: Learning rate (alpha+) for positive prediction errors
            negative_learning_rate: Learning rate (alpha-) for negative prediction errors
            initial_value: Initial Q-value for all machines
        """
        super().__init__(learning_rate=positive_learning_rate, initial_value=initial_value)
        self.positive_learning_rate = positive_learning_rate
        self.negative_learning_rate = negative_learning_rate
    
    def update(self, machine_id: str, reward: float) -> float:
        """
        Update the Q-value based on received reward using asymmetric learning rates.
        
        Args:
            machine_id: Identifier for the machine
            reward: Received reward
            
        Returns:
            Prediction error
        """
        self.initialize_machine(machine_id)
        current_value = self.q_values[machine_id]
        prediction_error = reward - current_value
        
        # Use appropriate learning rate based on prediction error sign
        if prediction_error >= 0:
            learning_rate = self.positive_learning_rate
        else:
            learning_rate = self.negative_learning_rate
            
        self.q_values[machine_id] += learning_rate * prediction_error
        return prediction_error
    
    def get_params(self) -> Dict[str, float]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameter values
        """
        return {
            "positive_learning_rate": self.positive_learning_rate,
            "negative_learning_rate": self.negative_learning_rate,
            "initial_value": self.initial_value
        }


class ThreeAlphaModel(RWPlusMinus):
    """
    Extended RW± model with three learning rates: separate rates for positive and negative
    prediction errors in free-choice trials, and a single rate for forced-choice trials.
    """
    
    def __init__(self, 
                 free_positive_learning_rate: float = 0.1, 
                 free_negative_learning_rate: float = 0.1,
                 forced_learning_rate: float = 0.1,
                 initial_value: float = 0.5):
        """
        Initialize the 3α model.
        
        Args:
            free_positive_learning_rate: Learning rate for positive prediction errors in free-choice trials
            free_negative_learning_rate: Learning rate for negative prediction errors in free-choice trials
            forced_learning_rate: Learning rate for all prediction errors in forced-choice trials
            initial_value: Initial Q-value for all machines
        """
        super().__init__(positive_learning_rate=free_positive_learning_rate,
                         negative_learning_rate=free_negative_learning_rate,
                         initial_value=initial_value)
        self.free_positive_learning_rate = free_positive_learning_rate
        self.free_negative_learning_rate = free_negative_learning_rate
        self.forced_learning_rate = forced_learning_rate
    
    def update(self, machine_id: str, reward: float, is_free_choice: bool = True) -> float:
        """
        Update the Q-value based on received reward using appropriate learning rates.
        
        Args:
            machine_id: Identifier for the machine
            reward: Received reward
            is_free_choice: Whether this update is from a free-choice trial (vs. forced)
            
        Returns:
            Prediction error
        """
        self.initialize_machine(machine_id)
        current_value = self.q_values[machine_id]
        prediction_error = reward - current_value
        
        if is_free_choice:
            # Use asymmetric learning rates for free-choice trials
            if prediction_error >= 0:
                learning_rate = self.free_positive_learning_rate
            else:
                learning_rate = self.free_negative_learning_rate
        else:
            # Use symmetric learning rate for forced-choice trials
            learning_rate = self.forced_learning_rate
            
        self.q_values[machine_id] += learning_rate * prediction_error
        return prediction_error
    
    def get_params(self) -> Dict[str, float]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameter values
        """
        return {
            "free_positive_learning_rate": self.free_positive_learning_rate,
            "free_negative_learning_rate": self.free_negative_learning_rate,
            "forced_learning_rate": self.forced_learning_rate,
            "initial_value": self.initial_value
        } 