import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops  # Changed greycomatrix and graycoprops imports
from skimage.color import rgb2gray

def extract_hog_features(image):
    # Convert image to grayscale
    gray_image = rgb2gray(image)
    # Extract HOG features
    features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

def extract_color_histogram(image, bins=(8, 8, 8)):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp_features(image, P=8, R=1):
    # Convert image to grayscale
    gray_image = rgb2gray(image)
    # Compute LBP
    lbp = local_binary_pattern(gray_image, P, R, method='uniform')
    # Calculate histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def extract_glcm_features(image):
    # Convert image to grayscale
    gray_image = rgb2gray(image)
    # Create GLCM
    glcm = graycomatrix((gray_image * 255).astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
    # Calculate contrast, dissimilarity, homogeneity, energy, correlation
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_features(image):
    hog_feats = extract_hog_features(image)
    color_feats = extract_color_histogram(image)
    lbp_feats = extract_lbp_features(image)
    glcm_feats = extract_glcm_features(image)
    return np.concatenate([hog_feats, color_feats, lbp_feats, glcm_feats])

def extract_features_from_images(images):
    features = []
    for image in images:
        feats = extract_features(image)
        features.append(feats)
    return np.array(features)
