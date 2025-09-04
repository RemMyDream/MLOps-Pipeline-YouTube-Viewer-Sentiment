import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from helpers import create_logger, get_root_directory, load_params
import os
from scipy.sparse import load_npz
import pickle
import optuna
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import save_npz, csr_matrix
import scipy
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

# Logger_Configure
logger =  create_logger("model_building", "./log/model_building.log")

def load_data(data_path : str) -> tuple:
    # Load preprocessing_data
    try:
        x_train = load_npz(os.path.join(data_path, "x_train.npz"))
        y_train = np.load(os.path.join(data_path, "y_train.npy"))
        logger.debug("Read data successfully %s", data_path)    
        return x_train, y_train   
    except Exception as e:
        logger.error("Unexpected error when loading data %s", e)
        raise

def handling_imbalance(x_train: scipy.sparse.csr_matrix, y_train: pd.Series) -> tuple:
    try:
        adasyn = ADASYN(random_state=42)
        x_resampled, y_resampled = adasyn.fit_resample(x_train, y_train)
        logger.debug("Handled Imbalance by ADASYN")
        return csr_matrix(x_resampled), y_resampled
    except Exception as e:
        logger.error("Error when handling imbalance %s", e)
        raise

def train_model(x_train, y_train) -> tuple:
    def objective_xgboost(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "eval_metric": "mlogloss",
            "num_class":3
        }
        
        model = XGBClassifier(**params)
        pipeline = Pipeline([
            ("ADASYN", ADASYN(random_state=42)),
            ("model", model)
            ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(pipeline, x_train, y_train, cv = cv)
        return score.mean()

    def run_optuna_experiment():
        params = load_params("./params.yaml", logger)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_xgboost, n_trials=params['model_building']['n_trials'])

        best_params = study.best_params
        best_model = XGBClassifier(**best_params)
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_optimization_history(study).show()
        return best_params, best_model

    try:
        best_params, best_model = run_optuna_experiment()
        best_model.fit(x_train, y_train)
        logger.debug("Training XGBoost successfully !!")
        return best_params, best_model
    except Exception as e:
        logger.error("Error when training XGBoost %s", e)
        raise

def save_model(model, file_path: str) -> None:
    # Save the training model
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error when saving model to %s", file_path)
        raise

def main():
    root_dir = get_root_directory()
    try:
        x_train, y_train = load_data("./data/interim")
        best_params, best_model = train_model(x_train, y_train)
        best_params.update({"eval_metric": "mlogloss", "num_class": 3})
        save_model(best_model, "./artifact/xgboost_model.pkl")
        print(best_params)
        logger.debug("Done model building")
    except Exception as e:
        logger.error("Failed to complete model building process: %s", e)

if __name__ == "__main__":
    main()




