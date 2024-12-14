from sklearn.svm import SVC

def train_svm(X_train, y_train):
    """
    Train an SVM model.
    
    Args:
    - X_train (numpy array): Training feature vectors.
    - y_train (numpy array): Training labels.
    
    Returns:
    - model (SVC): Trained SVM model.
    """
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return model
