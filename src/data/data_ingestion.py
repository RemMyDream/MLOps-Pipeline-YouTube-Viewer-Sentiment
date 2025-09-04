from helpers import create_logger, load_params, get_root_directory
import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

# Logger Configure
logger = create_logger('data_ingestion', "./log/data_ingestion.log")

def load_data(url: str) -> pd.DataFrame:
    # Read data from csv
    try:
        df = pd.read_csv(url, header = 0, delimiter=',', names = ['comment', 'category'])
        logger.debug("Read data from %s", url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error when parsing csv file %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected when loading data %s", e)
        raise

def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    # Raw Preprocessing Data by handing missing values, duplicates
    try:
        df.dropna(inplace = True)
        df.drop_duplicates(inplace=True)
        df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
        df = df.dropna(subset=['category'])
        df = df[df['comment'].str.strip() != '']
        logger.debug("Raw preprocessing completed")
        return df
    except Exception as e:
        logger.error("Unexpected error when preprocessing data %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index = False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index = False)
        logger.debug("Save data successfully")
    except Exception as e:
        logger.error("Unexpected error when saving data %s", e)
        raise
def main():
    try:
        root_dir = "./"
        params = load_params(os.path.join(root_dir, "params.yaml"), logger)
        test_size = params['data_ingestion']['test_size']
        df = load_data("https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
        df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path=os.path.join(root_dir, "data"))
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()