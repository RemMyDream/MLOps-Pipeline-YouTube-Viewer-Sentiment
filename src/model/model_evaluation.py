import numpy as np
import pandas as pd
from helpers import create_logger, config_aws
import os
from scipy.sparse import load_npz
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import json
from mlflow.models import infer_signature
import boto3

# Configure logger
logger = create_logger("model_evaluation", "./log/model_evaluation.log")

def load_model(model_path: str):
    # Load the trained model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model
        logger.debug("Model loaded from %s", model_path)
    except Exception as e:
        logger.error("Error when loading model: %s", e)
        raise

def load_vectorizer(vectorizer_path: str):
    # Load the trained model
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
            return vectorizer
        logger.debug("Vectorizer loaded from %s", vectorizer_path)
    except Exception as e:
        logger.error("Error when loading vectorizer: %s", e)
        raise

def load_data(data_path : str) -> tuple:
    # Load preprocessing_data
    try:
        x_test = load_npz(os.path.join(data_path, "x_test.npz"))
        y_test = np.load(os.path.join(data_path, "y_test.npy"))
        logger.debug("Read data successfully %s", data_path)        
        return x_test, y_test   
    except Exception as e:
        logger.error("Unexpected error when loading data %s", e)
        raise

def evaluate_model(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluation completed")
        return report, cm
    except Exception as e:
        logger.error("Error when evaluating model: %s", e)
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'./artifact/confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(experiment_id: str, run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'experiment_id': experiment_id,
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    config_aws("./config.yaml", logger)
    s3 = boto3.client("s3")
    print(s3.list_buckets())
    mlflow.set_tracking_uri("http://ec2-204-236-162-140.us-west-1.compute.amazonaws.com:5000/")
    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:
        try:
            # Load model and vectorizer:
            model = load_model("./artifact/xgboost_model.pkl")
            vectorizer = load_vectorizer("./artifact/tfidf_vectorizer.pkl")
            # Load test data
            x_test, y_test = load_data("./data/interim")

            # Validate the schema of input and output 
            input_example = pd.DataFrame(x_test.toarray()[:10],
                                         columns = vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(x_test[:10]))
            mlflow.sklearn.log_model(model, 
                                     "xgboost_model",
                                     signature = signature,
                                     input_example = input_example)
            
            artifact_uri = mlflow.get_artifact_uri()
            model_path = "xgboost_model"

            save_model_info(run.info.experiment_id, run.info.run_id, model_path, "./artifact/experiment_info.json")
            mlflow.log_artifact("./artifact/tfidf_vectorizer.pkl")
            report, cm = evaluate_model(model, x_test, y_test)
            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("mlflow.runName", "XGBoost Evaluation")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()





