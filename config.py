"""
Simple configuration loader for KRX stock data system.
Loads configuration from config.json file.
"""

import json
import logging
from typing import Dict, Any

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file has invalid JSON
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def get_api_key(config: Dict[str, Any]) -> str:
    """
    Get API key from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: API key
    """
    return config.get('api', {}).get('auth_key', '')

def get_database_path(config: Dict[str, Any]) -> str:
    """
    Get database path from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Database path
    """
    return config.get('database', {}).get('path', 'krx_stock_data.db')

def get_request_delay(config: Dict[str, Any]) -> float:
    """
    Get request delay from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        float: Request delay in seconds
    """
    return config.get('api', {}).get('request_delay', 1.0)