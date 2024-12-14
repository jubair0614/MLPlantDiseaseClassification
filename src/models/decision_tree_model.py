from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree model.
    
    Args:
    - X_train (numpy array): Training feature vectors.
    - y_train (numpy array): Training labels.
    
    Returns:
    - model (DecisionTreeClassifier): Trained Decision Tree model.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model
