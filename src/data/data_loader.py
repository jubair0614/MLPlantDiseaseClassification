import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_sample_data(sample_dir):
    images = []
    labels = []

    for img_name in os.listdir(sample_dir):
        img_path = os.path.join(sample_dir, img_name)
        
        img = cv2.imread(img_path)  
        
        if img is not None:
            img = cv2.resize(img, (256, 256))  # Resize to 256x256
            images.append(img)
            
            # Extract the label from the filename (e.g., 'tomato_early_blight.jpg')
            if '_' in img_name:
                label = img_name.split('_', 1)[1].rsplit('.', 1)[0]
                labels.append(label)
            else:
                labels.append('unknown')  # In case there is no underscore

    return np.array(images), labels

def process_label(label):
    """
    Process the label to remove underscores and convert it to Pascal case.
    
    Args:
    - label (str): The original label string (e.g., 'Early_blight').

    Returns:
    - str: The processed label in Pascal case (e.g., 'EarlyBlight').
    """
    # Remove any leading underscores and split by underscores
    parts = label.lstrip('_').split('_')
    
    # Convert to Pascal case
    return ''.join(word.capitalize() for word in parts)

def load_plantVillage_data(sample_dir):
    """
    Load images from the PlantVillage dataset directory, and create separate datasets
    for Tomato, Pepper, and Potato based on folder names.
    
    Args:
    - sample_dir (str): The root directory containing 'PlantVillage' folders.

    Returns:
    - datasets (dict): A dictionary containing image data and labels for tomato, pepper, and potato.
        Example:
        {
            'tomato': (tomato_images, tomato_labels),
            'pepper': (pepper_images, pepper_labels),
            'potato': (potato_images, potato_labels)
        }
    """
    
    datasets = {
        'tomato': ([], []),
        'pepper': ([], []),
        'potato': ([], [])
    }

    for folder_name in os.listdir(sample_dir):
        folder_path = os.path.join(sample_dir, folder_name)
        
        # Check if it's a folder and starts with Tomato_, Pepper_, or Potato_
        if os.path.isdir(folder_path):
            if folder_name.startswith('Tomato_'):
                crop_type = 'tomato'
            elif folder_name.startswith('Pepper_'):
                crop_type = 'pepper'
            elif folder_name.startswith('Potato_'):
                crop_type = 'potato'
            else:
                continue

            # Extract class label (everything after the underscore)
            label = folder_name.split('_', 1)[1]
            label = process_label(label)

            # Load all images from this folder
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (256, 256))
                        datasets[crop_type][0].append(img_resized)
                        datasets[crop_type][1].append(label)
                    else:
                        print(f"Failed to load image: {img_path}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    # Convert lists to numpy arrays
    for crop_type in datasets:
        datasets[crop_type] = (np.array(datasets[crop_type][0]), datasets[crop_type][1])
    
    return datasets


def load_potato_data(sample_dir):
    """
    Load images from the PlantVillage dataset directory for Potato crops.
    
    Args:
    - sample_dir (str): The root directory containing the 'PlantVillage' folders.

    Returns:
    - potato_images (np.array): Array containing potato images.
    - potato_labels (list): List containing corresponding labels for potato images.
    """

    potato_images = []
    potato_labels = []

    for folder_name in os.listdir(sample_dir):
        folder_path = os.path.join(sample_dir, folder_name)

        # Check if it's a folder and starts with Potato_
        if os.path.isdir(folder_path) and folder_name.startswith('Potato_'):
            # Extract class label (everything after the underscore)
            label = folder_name.split('_', 1)[1]
            label = process_label(label)

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (256, 256))
                        potato_images.append(img_resized)
                        potato_labels.append(label)
                    else:
                        print(f"Failed to load image: {img_path}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # Convert lists to numpy array
    potato_images = np.array(potato_images)
    
    return potato_images, potato_labels


def load_tomato_data(sample_dir):
    """
    Load images from the PlantVillage dataset directory for Potato crops.
    
    Args:
    - sample_dir (str): The root directory containing the 'PlantVillage' folders.

    Returns:
    - potato_images (np.array): Array containing potato images.
    - potato_labels (list): List containing corresponding labels for potato images.
    """

    tomato_images = []
    tomato_labels = []

    for folder_name in os.listdir(sample_dir):
        folder_path = os.path.join(sample_dir, folder_name)

        # Check if it's a folder and starts with Potato_
        if os.path.isdir(folder_path) and folder_name.startswith('Tomato_'):
            # Extract class label (everything after the underscore)
            label = folder_name.split('_', 1)[1]
            label = process_label(label)

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, (256, 256))
                        tomato_images.append(img_resized)
                        tomato_labels.append(label)
                    else:
                        print(f"Failed to load image: {img_path}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # Convert lists to numpy array
    tomato_images = np.array(tomato_images)
    
    return tomato_images, tomato_labels

def encode_lables(labels):
    label_encoder = LabelEncoder()

    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = np.array(labels_encoded)

    return labels_encoded