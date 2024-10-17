import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ..features.feature_extractor import extract_features

def train_random_forest(potato_images, potato_labels):
    # Extract features from potato images
    potato_features = extract_features(potato_images)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(potato_features, potato_labels, test_size=0.2, random_state=42)

    # Create and train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))
