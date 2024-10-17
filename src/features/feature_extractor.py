import cv2
import numpy as np
from skimage.feature import hog

def convert_to_grayscale(image):
    """Convert a color image (256, 256, 3) to grayscale (256, 256)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Function to calculate symmetry of the image
def calculate_symmetry(image):
    """Calculate symmetry feature of an image by comparing the original and flipped version."""
    
    # Convert to float32 to avoid overflow
    image = image.astype(np.float32)

    # Flip the image horizontally
    flip_image = np.flip(image, 1)

    # Calculate pixel-wise absolute difference
    diff = np.abs(image - flip_image)

    # Calculate normalized symmetry
    sys = -np.sum(diff) / image.size

    return sys

# Function to calculate intensity of the image
def calculate_intensity(image):
    """Calculate the average intensity of a grayscale image."""
    intense = np.sum(image) / image.size  # Normalized intensity measure
    return intense

def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    
    Args:
    - image (numpy array): Input image of shape (256, 256, 3).
    
    Returns:
    - hog_features (numpy array): HOG features of the image.
    """
    # Convert the image to grayscale for HOG
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define HOG parameters
    hog_features, hog_image = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    
    return hog_features

def extract_features_from_dataset(images, feature_type="hog"):
    """
    Extract features from a dataset of images.
    
    Args:
    - images (numpy array): Dataset of images.
    - feature_type (str): Type of feature extraction ('hog', etc.)
    
    Returns:
    - features (numpy array): Extracted feature vectors.
    """
    feature_list = []
    
    for image in images:
        if feature_type == "hog":
            features = extract_hog_features(image)
        else:
            raise ValueError("Unsupported feature type: {}".format(feature_type))
        
        feature_list.append(features)
    
    return np.array(feature_list)

def extract_features(images):
    features = []
    
    for img in images:
        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Calculate histogram
        hist = cv2.calcHist([hsv_img], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    
    return np.array(features)