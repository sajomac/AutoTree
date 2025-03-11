# src/main.py
import sys
import os
from preprocessing import *
from modeling import *
from visualization import *
import argparse
import pandas as pd
import joblib


# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the random forest pipeline.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset (CSV file)."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the target column."
    )
    parser.add_argument(
        "--categorical",
        action="store_true",
        help="Whether the target is categorical."
    )
    parser.add_argument(
        "--tar_skew",
        action="store_true",
        help="Whether to correct skewness for the target variable."
    )
    parser.add_argument(
        "--pred_skew",
        action="store_true",
        help="Whether to correct skewness for predictor variables."
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default="",
        help="Comma-separated list of columns to remove."
    )
    parser.add_argument(
        "--identify_predictors",
        action="store_true",
        help="Whether to display feature importances."
    )
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="Whether to generate visualizations."
    )
    parser.add_argument(
        "--dim_reduce",
        action="store_true",
        help="Whether to apply dimensionality reduction."
    )
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data_path)

    # Convert columns_to_remove to a list
    columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    print(args.categorical)
    # Run the pipeline
    results, model = random_forest(
        data,
        tar=args.target,
        categorical=args.categorical,
        tar_skew=args.tar_skew,
        pred_skew=args.pred_skew,
        columns_to_remove=columns_to_remove,
        identify_predictors=args.identify_predictors,
        graphs=args.graphs,
        dim_reduce=args.dim_reduce
    )

    # Create a directory to save the model
    model_dir = "generated_model"
    os.makedirs(model_dir, exist_ok=True)

    # Generate a descriptive name for the model file
    model_name = f"model_{args.target}_{'categorical' if args.categorical else 'continuous'}.joblib"
    model_path = os.path.join(model_dir, model_name)

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()