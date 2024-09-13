import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

def test_train_model():
    """Test if the train_model function returns a model of the expected type."""
    X_train = np.random.rand(100, 10)  # Example feature matrix
    y_train = np.random.randint(0, 2, 100)  # Example labels
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not of type RandomForestClassifier"

def test_compute_model_metrics():
    """Test if compute_model_metrics returns metrics of the expected type."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float), "Precision is not of type float"
    assert isinstance(recall, float), "Recall is not of type float"
    assert isinstance(fbeta, float), "F1 score is not of type float"

def test_process_data_output():
    """Test if process_data function returns expected output types."""
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'salary': ['<=50K', '>50K', '<=50K']
    })
    cat_features = ['feature1']
    label = 'salary'
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label=label, training=True)
    assert isinstance(X, np.ndarray), "X is not of type np.ndarray"
    assert isinstance(y, np.ndarray), "y is not of type np.ndarray"
    assert encoder is not None, "Encoder is None"
    assert lb is not None, "Label Binarizer is None"

def test_data_splitting():
    """Test if the train-test split results in the expected dataset sizes."""
    data = pd.read_csv('data/census.csv')
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    assert len(train) > 0, "Training dataset is empty"
    assert len(test) > 0, "Test dataset is empty"
    assert len(train) + len(test) == len(data), "Train and test dataset sizes do not sum to the original dataset size"
