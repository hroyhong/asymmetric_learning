"""
DeepSeek model API integration via Volcengine.

This module provides functions to interact with the DeepSeek-R1 model hosted on 
Volcengine's platform for the slot machine experiments.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekClient:
    """
    Client for interacting with the DeepSeek model via Volcengine.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "deepseek-r1-250120"):
        """
        Initialize the DeepSeek client.
        
        Args:
            api_key: API key for accessing the Volcengine API. If None, it will be read from 
                    the ARK_API_KEY environment variable.
            model_id: The model endpoint ID to use. Defaults to DeepSeek-R1.
        """
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as ARK_API_KEY environment variable")
        
        self.model_id = model_id
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        logger.info(f"DeepSeek client initialized with model ID: {model_id}")
    
    def get_slot_machine_choice(self, prompt: str) -> str:
        """
        Get the model's choice for a slot machine experiment.
        
        Args:
            prompt: The prompt describing the experiment and history
            
        Returns:
            The chosen machine (e.g., "A", "B", etc.)
        """
        logger.info("Sending prompt to DeepSeek model")
        try:
            # Send the request to the model
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are an AI making decisions in a slot machine experiment. Always respond with just the machine letter (e.g., 'A', 'B', etc.) without any explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Use deterministic responses for consistency
                max_tokens=10      # We only need a short response
            )
            
            # Extract the chosen machine from the response
            response = completion.choices[0].message.content.strip()
            logger.info(f"Model response: {response}")
            
            # Extract just the machine letter if there's additional text
            if "Machine " in response:
                machine = response.split("Machine ")[1].strip()
            else:
                machine = response.strip()
            
            logger.info(f"Extracted machine choice: {machine}")
            return machine
        
        except Exception as e:
            logger.error(f"Error getting response from DeepSeek model: {str(e)}")
            raise

def get_api_client() -> DeepSeekClient:
    """
    Factory function to get a configured DeepSeek client.
    
    Returns:
        Configured DeepSeek client instance
    """
    return DeepSeekClient() 