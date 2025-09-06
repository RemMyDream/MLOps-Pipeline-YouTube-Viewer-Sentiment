import numpy as np
import pandas as pd
from helpers import create_logger, config_aws
import os
from scipy.sparse import load_npz
import pickle
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import json
from mlflow.models import infer_signature
import boto3


logger = create_logger("model_registration", "./log/model_registration.log")

def load_model_info(model_path: str) -> dict:
    try:
        with open(model_path, "r") as f:
            model_info = json.load(f)
        logger.debug('Loading model info successfully')
        return model_info
    except FileNotFoundError:
        logger.error("Not found %s", model_path)
        raise
    except Exception as e:
        logger.error("Error when loading model info: %s", e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        # Register model for managing version, stage
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version} is registered and transitioned to Staging stage")
    except Exception as e:
        logger.error("Error when register model: %s", e)
        raise

def main():
    config_aws("./config.yaml", logger)
    try:
        mlflow.set_tracking_uri("http://ec2-204-236-162-140.us-west-1.compute.amazonaws.com:5000/")
        model_info = load_model_info("./artifact/experiment_info.json")
        register_model("youtube_xgboost_model", model_info)
    except Exception as e:
        logger.error("Failed to complete registration process: %s", e)

if __name__ == "__main__":
    main()

