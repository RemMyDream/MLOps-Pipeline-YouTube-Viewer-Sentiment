import numpy as np
import pandas as pd
import os
import pickle
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import save_npz, csr_matrix
import scipy
from helpers import create_logger, load_params, get_root_directory
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = create_logger('data_preprocessing', './log/data_preprocess.log')

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['comment'] = df['comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, ngram_range: tuple, vectorizer_max_features: int) -> tuple:
    # Apply TF-IDF to data
    try:
        vec = TfidfVectorizer(ngram_range = ngram_range, max_features = vectorizer_max_features)

        x_train = train_data['comment']
        y_train = train_data['category']
        x_test=  test_data['comment']
        y_test = test_data['category']
        x_train = vec.fit_transform (x_train)
        x_test = vec.transform(x_test)        
        logger.debug(f"Apply TF-IDF successfully. Train shape {x_train.shape}")
        with open("./artifact/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vec, f)
            logger.debug("Save vectorizer successfully")
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logger.error("Unexpected Error when applying tfidf %s", e)
        raise

def save_data(x_train: scipy.sparse.csr_matrix, x_test: scipy.sparse.csr_matrix, y_train: pd.Series, y_test: pd.Series, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        save_npz(os.path.join(interim_data_path, "x_train.npz"), x_train)
        save_npz(os.path.join(interim_data_path, "x_test.npz"), x_test)
        np.save(os.path.join(interim_data_path, "y_train.npy"), y_train.to_numpy())
        np.save(os.path.join(interim_data_path, "y_test.npy"), y_test.to_numpy())

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        root_dir = "./"
        logger.debug("Starting data preprocessing...")
        params = load_params(os.path.join("params.yaml"), logger)
        # Fetch the data from data/raw
        train_data = pd.read_csv(os.path.join(root_dir, 'data/raw/train.csv'))
        test_data = pd.read_csv(os.path.join(root_dir, 'data/raw/test.csv'))
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        x_train, x_test, y_train, y_test = apply_tfidf(train_processed_data, test_processed_data, ngram_range=tuple(params['tf_idf']['ngram_range']), vectorizer_max_features=params['tf_idf']['vectorizer_max_features'])

        # Save the processed data
        save_data(x_train, x_test, y_train, y_test, data_path=os.path.join(root_dir, "data"))
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()