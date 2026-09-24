import yaml
from src.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """load the YAML config file into a dict"""
    logger.info(f'Loading config from {config_path}')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config