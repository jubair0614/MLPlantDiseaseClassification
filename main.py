from src.data.data_loader import load_sample_data
from src.data.data_loader import load_plantVillage_data
from src.features.feature_extractor import extract_hog_features
from src.models.random_forest_model import train_random_forest

def main():
    # Define the path to the sample images directory
    sample_dir = 'data/samples'
    
    # Load the sample data
    print("Loading sample data...")
    sample_images, sample_labels = load_sample_data(sample_dir)
    
    # Display the results
    print(f"Loaded {len(sample_images)} sample images.")
    print("Sample Labels:")
    for label in sample_labels:
        print(label)

    explore_dataset()
    print(extract_hog_features(sample_images[0]))


def explore_dataset():
    dir = "data/PlantVillage"
    datasets = load_plantVillage_data(dir)

    # Access tomato dataset
    tomato_images, tomato_labels = datasets['tomato']

    # Access pepper dataset
    pepper_images, pepper_labels = datasets['pepper']

    # Access potato dataset
    potato_images, potato_labels = datasets['potato']

    details(tomato_images, tomato_labels, "tomato")
    details(pepper_images, pepper_labels, "pepper")
    details(potato_images, potato_labels, "potato")

    train_random_forest(potato_images, potato_labels)

def details(images, labels, leaf_name):
    # Print the number of images and labels
    print(f"Number of {leaf_name} images: {len(images)}")
    print(f"Number of {leaf_name} labels: {len(labels)}")

    # Print the shape of the first image in the dataset (height, width, channels)
    if len(images) > 0:
        print(f"Shape of first {leaf_name} image: {images[0].shape}")

    # Print the unique labels
    unique_labels = set(labels)
    print(f"Unique {leaf_name} labels: {unique_labels}")

    # Check if the number of images and labels match
    if len(images) == len(labels):
        print("The number of images matches the number of labels.")
    else:
        print("Mismatch between the number of images and labels!")

if __name__ == "__main__":
    main()