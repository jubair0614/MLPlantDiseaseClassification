from sklearn.model_selection import KFold
import numpy as np

def split_data(features, labels, n_splits=5):
    """
    Split data into training and validation sets using KFold cross-validation.
    
    Args:
    - features (numpy array): Extracted feature vectors.
    - labels (numpy array): Encoded labels.
    - n_splits (int): Number of folds for cross-validation.
    
    Returns:
    - generator: Yields training and validation sets for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(features):
        X_train, X_val = features[train_index], features[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        yield X_train, X_val, y_train, y_val
