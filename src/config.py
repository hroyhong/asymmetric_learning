"""
Configuration parameters for the asymmetric learning experiments.
"""

# General configuration
SEED = 42
OUTPUT_DIR = "results"
DATA_DIR = "data"

# Task 1: Basic Partial Feedback
TASK1_CONFIG = {
    "name": "basic_partial_feedback",
    "n_casinos": 4,
    "n_visits_per_casino": 24,
    "total_visits": 96,
    "reward_values": [0.5, 0.0],
    "reward_probabilities": [
        (0.25, 0.25),  # (machine1, machine2) - equal low probabilities
        (0.25, 0.75),  # low vs high
        (0.75, 0.25),  # high vs low
        (0.75, 0.75),  # equal high probabilities
    ],
    "trials_per_block": 40,
}

# Task 2: Full Feedback
TASK2_CONFIG = {
    "name": "full_feedback",
    "n_visits": 40,
    "reward_values": [1.0, -1.0],
    "reward_probabilities": [
        (0.9, 0.6),    # high-reward block
        (0.4, 0.1),    # low-reward block
    ],
    "trials_per_block": 40,
}

# Task 3: Agency Condition
TASK3_CONFIG = {
    "name": "agency_condition",
    "free_choice_config": {
        "n_visits": 40,
        "reward_values": [1.0, -1.0],
        "trials_per_block": 40,
    },
    "mixed_choice_config": {
        "total_visits": 80,
        "free_choice_visits": 40,
        "forced_choice_visits": 40,
        "reward_values": [1.0, -1.0],
        "trials_per_block": 80,
    },
    "reward_probabilities": [
        (0.7, 0.3),
        (0.3, 0.7),
    ]
}

# Model fitting parameters
MODEL_CONFIG = {
    "rw": {  # Standard Rescorla-Wagner model
        "learning_rate_range": (0.01, 1.0),
        "initial_value_range": (0.0, 1.0),
    },
    "rw_plus_minus": {  # Asymmetric RW model with separate learning rates
        "positive_learning_rate_range": (0.01, 1.0),
        "negative_learning_rate_range": (0.01, 1.0),
        "initial_value_range": (0.0, 1.0),
    },
    "three_alpha": {  # 3-alpha model for free/forced choice
        "free_positive_learning_rate_range": (0.01, 1.0),
        "free_negative_learning_rate_range": (0.01, 1.0),
        "forced_learning_rate_range": (0.01, 1.0),
        "initial_value_range": (0.0, 1.0),
    }
}

# Analysis parameters
ANALYSIS_CONFIG = {
    "n_bootstraps": 1000,
    "confidence_interval": 0.95,
    "n_simulations": 100,
} 