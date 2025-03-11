# AutoTree: Automated Machine Learning Pipeline for Regression and Classification

AutoTree is a Python-based machine learning pipeline that automates the training and evaluation of regression and classification models using HistGradientBoostingRegressor and HistGradientBoostingClassifier from scikit-learn. It automatically determines reasonable hyperparameter options based on the provided dataset and performs a grid search to identify the optimal set of hyperparameters. The pipeline streamlines data preprocessing, feature engineering, model training, evaluation, visualization, and model saving.

All of this can be done with a single command. See the Tree_Example.ipynb for performance accross iris, diabetes, titanic, digits, california_housing, wine and breast_cancer datasets.

## Features

### Data Preprocessing
- Handles missing values
- Corrects skewness in data
- Removes collinear columns
- Encodes categorical variables

### Model Training
- Supports both regression and classification tasks
- Uses Lasso in the event of overfitting
- Iteratively reduces model complexity and PCA retained variance if overfitting persists

### Feature Importance
- Provides permutation-based feature importance analysis

### Visualization
- Generates plots for model evaluation, including:
  - Actual vs. predicted values (for regression)
  - Confusion matrices and ROC curves (for classification)
  - PCA clustering plots (when dimensionality reduction is applied)

### Dimensionality Reduction
- Automatically applies PCA and Lasso-based feature selection in case of overfitting

## Installation

### Requirements
- Python 3.12 or higher
- pip (Python package manager)

### Installation Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/auto-tree.git
   cd auto-tree
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script with sample data:
   ```sh
   ./run_pipeline.sh
   ```
4. For more comprehensive examples and use cases:
   ```sh
   python src/main_example.py
   ```
   You can also use the provided Jupyter Notebook: `Tree_Example.ipynb`

## Usage

1) You can run the pipeline using a shell script. Modify `run_pipeline.sh` to specify your dataset and parameters:

```sh
# Modify these parameters in run_pipeline.sh
DATA_PATH="/path/to/your/data.csv"
TARGET="your_target_column"
CATEGORICAL=false               # Set to true for classification tasks
TAR_SKEW=true                   # Apply transformation if the target column is skewed
PRED_SKEW=true                  # Apply transformation if predictors are skewed
COLUMNS_TO_REMOVE=""            # Specify columns to remove (e.g., "a,b,c,d")
IDENTIFY_PREDICTORS=true        # Generate a predictor hierarchy
GRAPHS=true                     # Enable graphical output for predictors, PCA, and predictions
DIM_REDUCE=true                 # Enable Lasso feature selection by default
```

Make the script executable and run it:
```sh
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Example Datasets
Example datasets are available in `./src/example_main.py`.

## Output

The pipeline generates the following outputs:

### Model Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² score
- F1 score (for classification)

### Saved Model
- The trained model is saved in: `auto_tree/generated_model`

### Visualizations
- **Regression:** Actual vs. Predicted values
- **Classification:** Confusion matrix and ROC curve
- **Feature Analysis:** Feature importance plots, PCA clustering (if applied)

## Project Structure
```
auto-tree/
│
├── src/                    # Source code
│   ├── __init__.py         # Marks the folder as a package
│   ├── main.py             # Entry point for the pipeline
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── modeling.py         # Model training and evaluation
│   ├── visualization.py    # Visualization functions
│
├── data/                   # Folder for datasets (optional)
│   └── your_dataset.csv
│
├── run_pipeline.sh         # Shell script for running the pipeline
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
└── .gitignore              # Files to ignore
```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```sh
   git checkout -b feature/YourFeatureName
   ```
3. Commit your changes:
   ```sh
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```sh
   git push origin feature/YourFeatureName
   ```
5. Open a pull request.

---

For any issues or feature requests, feel free to create an issue in the repository!

