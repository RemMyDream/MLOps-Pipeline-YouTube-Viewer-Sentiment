import logging
import os
import yaml
import mlflow

def create_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # File
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_params(params_path:str, logger) -> dict:
    # Load params from files
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
            logger.debug("Reading params successfully")
            return params
    except FileNotFoundError:
        logger.error("Not found params_path %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error ̀̀̀%s", e)
        raise
    except Exception as e:
        logger.error("Unexpected Error when loading param %s", e)
        raise

def config_aws(config_path: str, logger):
    params = load_params(config_path, logger)
    os.environ['AWS_ACCESS_KEY_ID'] = params['aws']['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = params['aws']['aws_secret_access_key']
    os.environ['REGION'] = params['aws']['region']

