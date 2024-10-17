# Plant Disease Classification

This project aims to classify plant diseases using images of leaves from the PlantVillage dataset. The current implementation focuses on identifying diseases in potato leaves, but the code is structured to allow for easy extension to other crops, such as tomatoes and peppers.

## Project Structure

```
MLPlantDiseaseClassification/
├── data/
│   └── PlantVillage/
│       ├── Potato_healthy/
│       ├── Potato_Early_blight/
│       └── Potato_Late_blight/
├── notebooks/
│   └── dataExplore.ipynb
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── features/
│   │   └── feature_extractor.py
│   └── models/
│       └── random_forest_model.py
├── .gitignore
├── env.yml
└── README.md
```

## Prerequisites

Make sure you have the following software installed on your machine:

- [Anaconda](https://www.anaconda.com/products/distribution) (recommended)
- Git (optional, for version control)

## Setting Up the Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/jubair0614/MLPlantDiseaseClassification.git
   cd MLPlantDiseaseClassification
   ```

2. Create a conda environment from the `env.yml` file:

   ```bash
   conda env create -f env.yml
   ```

3. Activate the conda environment:

   ```bash
   conda activate plant-leaf-disease-classification
   ```

## Running the Project

Once the environment is set up and activated, you can run the main script. 

1. Navigate to the `src/models/` directory:

   ```bash
   cd src/models/
   ```

2. Run the `random_forest_model.py` script:

   ```bash
   python random_forest_model.py
   ```

This will execute the model training and evaluation process using the potato leaf images from the PlantVillage dataset.

## Additional Notes

- Ensure that you have downloaded the PlantVillage dataset and organized it in the specified directory structure under `data/PlantVillage/`.
- The dataset folders should contain images organized in subfolders named after the disease classes (e.g., `Potato_healthy`, `Potato_Early_blight`, `Potato_Late_blight`).

## Contribution

Feel free to contribute to this project! If you'd like to add features or fix issues, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
