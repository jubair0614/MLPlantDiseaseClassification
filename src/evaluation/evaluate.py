from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_val, y_val):
    """
    Evaluate a model using accuracy, precision, recall, and F1-score.
    
    Args:
    - model: Trained machine learning model.
    - X_val (numpy array): Validation feature vectors.
    - y_val (numpy array): Validation labels.
    
    Returns:
    - dict: Dictionary containing the evaluation metrics.
    """
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
