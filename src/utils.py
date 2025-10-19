import logging
import os
import random
import time
from functools import wraps
from typing import Any, Dict

import numpy as np
import yaml
from yaml import FullLoader


def setup_logger() -> logging.Logger:
    """Set up a basic logger that prints to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def seed_everything(seed: int = 42):
    """
    Set seeds for reproducibility across all relevant libraries.

    Args:
        seed: The integer value to use as the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    logger.info(f"Seeds set to {seed} for random, os, and numpy.")


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary.

    Args:
        filepath: The path to the YAML file.

    Returns:
        A dictionary with the file's contents.
    """
    try:
        with open(filepath, "r") as f:
            return yaml.load(f, Loader=FullLoader)
    except FileNotFoundError:
        logger.error(f"YAML file not found at: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading YAML file '{filepath}': {e}")
        raise


def save_yaml(filepath: str, data: Dict[str, Any]):
    """
    Save a dictionary to a YAML file.

    Args:
        filepath: The path to save the YAML file to.
        data: The dictionary to save.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving YAML file to '{filepath}': {e}")
        raise


def timer(func):
    """A decorator to time the execution of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {run_time:.4f} seconds")
        return result

    return wrapper
