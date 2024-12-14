from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    
    Args:
    - X_train (numpy array): Training feature vectors.
    - y_train (numpy array): Training labels.
    
    Returns:
    - model (RandomForestClassifier): Trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
