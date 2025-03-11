import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from preprocessing import *
from modeling import *
from visualization import *




OVERFIT_THRESHOLD = 0.1
MIN_SAMPLES_LEAF = 20
VALIDATION_FRACTION = 0.2
MAX_DEPTH = 20
LEARNING_RATE = 0.1
L2_REGULARIZATION = 0.1
TOL = 1e-5

from sklearn.model_selection import GridSearchCV

def reduce_regression_complexity(overall_mae, mean_test_mae, overall_r2, mean_test_r2, mse_list, mae_list, r2_list, X, y, var_to_retain, X_scaled, kf, MAX_ITER):
    """
    Reduces the complexity of a regression model by iteratively adjusting PCA components and model hyperparameters.

    This function aims to reduce overfitting by iteratively decreasing the number of PCA components and adjusting
    hyperparameters of a HistGradientBoostingRegressor model. It evaluates the model's performance using cross-validation
    and stops when the difference between overall metrics and mean test metrics falls below a certain threshold or
    after a maximum number of iterations.

    Parameters:
    - overall_mae (float): Overall Mean Absolute Error before reduction.
    - mean_test_mae (float): Mean Test Mean Absolute Error before reduction.
    - overall_r2 (float): Overall R² score before reduction.
    - mean_test_r2 (float): Mean Test R² score before reduction.
    - mse_list (list): List to store Mean Squared Error for each fold.
    - mae_list (list): List to store Mean Absolute Error for each fold.
    - r2_list (list): List to store R² scores for each fold.
    - X (DataFrame): Feature set.
    - y (Series): Target variable.
    - var_to_retain (int): Initial number of PCA components to retain.
    - X_scaled (array-like): Scaled feature set.
    - kf (KFold): K-Folds cross-validator.
    - MAX_ITER (int): Maximum number of iterations for the model.

    Returns:
    - mean_test_mse (float): Mean Test Mean Squared Error after reduction.
    - mean_test_mae (float): Mean Test Mean Absolute Error after reduction.
    - mean_test_r2 (float): Mean Test R² score after reduction.
    - overall_mse (float): Overall Mean Squared Error after reduction.
    - overall_mae (float): Overall Mean Absolute Error after reduction.
    - overall_r2 (float): Overall R² score after reduction.
    - X (DataFrame): Transformed feature set after PCA.
    - var_to_retain (int): Number of PCA components retained after reduction.
    - X_scaled (array-like): Scaled feature set after reduction.
    - best_model (HistGradientBoostingRegressor): Best model found during grid search.
    - pca (PCA): PCA object used for dimensionality reduction.
    """
    i = 0
    while ((abs((overall_mae - mean_test_mae) / mean_test_mae) > OVERFIT_THRESHOLD or
            (overall_r2 - mean_test_r2) / mean_test_r2 > OVERFIT_THRESHOLD) and i < 10):
        print("Reduction in model complexity, iteration:", i)
        for train_index, test_index in kf.split(X):
            # Adjust PCA component retention for each iteration
            pca = PCA(n_components=var_to_retain - i / 20)
            X_pca = pca.fit_transform(X_scaled)
            X = pd.DataFrame(X_pca, columns=[f'PCA_{j+1}' for j in range(X_pca.shape[1])])

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            n_samples = len(X_train)
            n_features = len(X_train.columns)

            # Define the parameter grid
            param_grid = {
                'max_iter': [int(min(MAX_ITER, int(0.1 * n_samples) - i * int(0.1 * n_samples) / 10))],
                'learning_rate': [LEARNING_RATE, 0.05, 0.01],
                'max_depth': [int(min(MAX_DEPTH, int(np.log2(n_features) - i * (np.log2(n_features) / 10))) if n_features > 0 else 10)],
                'min_samples_leaf': [int(max(MIN_SAMPLES_LEAF, int(0.01 * n_samples) - i * int(0.01 * n_samples) / 10))],
                'l2_regularization': [L2_REGULARIZATION],
                'validation_fraction': [min(VALIDATION_FRACTION, 1000 / n_samples - i / 10 * (1000 / n_samples))],
                'tol': [TOL]
            }

            # Initialize the model
            model = HistGradientBoostingRegressor(
                early_stopping=True,
                scoring="neg_mean_squared_error"
            )

            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Best model from grid search
            best_model = grid_search.best_estimator_

            # Predict and evaluate performance for current fold
            y_test_pred = best_model.predict(X_test)
            mse_list.append(mean_squared_error(y_test, y_test_pred))
            mae_list.append(mean_absolute_error(y_test, y_test_pred))
            r2_list.append(stats.pearsonr(y_test, y_test_pred)[0]**2)

        # Update metrics after dimensionality reduction
        mean_test_mse = np.mean(mse_list)
        mean_test_mae = np.mean(mae_list)
        mean_test_r2 = np.mean(r2_list)

        # Evaluate overall performance with reduced features
        y_pred_overall = best_model.predict(X)
        overall_mse = mean_squared_error(y, y_pred_overall)
        overall_mae = mean_absolute_error(y, y_pred_overall)
        overall_r2 = r2_score(y, y_pred_overall)

        print("\nMetrics after dimensionality reduction:")
        print(f"Mean test MSE: {mean_test_mse:.2f}")
        print(f"Mean test MAE: {mean_test_mae:.2f}")
        print(f"Mean test R²: {mean_test_r2:.2f}")
        i += 1

    return mean_test_mse, mean_test_mae, mean_test_r2, overall_mse, overall_mae, overall_r2, X, var_to_retain, X_scaled, best_model, pca


from sklearn.model_selection import GridSearchCV

def reduce_classification_complexity(overall_dice, mean_test_dice, dice, X, y, var_to_retain, X_scaled, kf, MAX_ITER):
    """
    Reduces the complexity of a classification model by iteratively adjusting PCA components and model hyperparameters.

    This function aims to reduce overfitting by iteratively decreasing the number of PCA components and adjusting
    hyperparameters of a HistGradientBoostingClassifier model. It evaluates the model's performance using cross-validation
    and stops when the difference between overall metrics and mean test metrics falls below a certain threshold or
    after a maximum number of iterations.

    Parameters:
    - overall_dice (float): Overall F1 score before reduction.
    - mean_test_dice (float): Mean Test F1 score before reduction.
    - dice (list): List to store F1 scores for each fold.
    - X (DataFrame): Feature set.
    - y (Series): Target variable.
    - var_to_retain (int): Initial number of PCA components to retain.
    - X_scaled (array-like): Scaled feature set.
    - kf (KFold): K-Folds cross-validator.
    - MAX_ITER (int): Maximum number of iterations for the model.

    Returns:
    - overall_dice (float): Overall F1 score after reduction.
    - mean_test_dice (float): Mean Test F1 score after reduction.
    - X (DataFrame): Transformed feature set after PCA.
    - var_to_retain (int): Number of PCA components retained after reduction.
    - X_scaled (array-like): Scaled feature set after reduction.
    - best_model (HistGradientBoostingClassifier): Best model found during grid search.
    - mean_test_dice (float): Mean Test F1 score after reduction.
    - overall_dice (float): Overall F1 score after reduction.
    - pca (PCA): PCA object used for dimensionality reduction.
    """
    i = 0
    while ((abs((overall_dice - mean_test_dice) / mean_test_dice) > OVERFIT_THRESHOLD) and i < 10):
        print("Reduction in model complexity, iteration:", i)
        for train_index, test_index in kf.split(X):
            # Adjust PCA component retention for each iteration
            pca = PCA(n_components=var_to_retain - i / 20)
            X_pca = pca.fit_transform(X_scaled)
            X = pd.DataFrame(X_pca, columns=[f'PCA_{j+1}' for j in range(X_pca.shape[1])])

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            n_samples = len(X_train)
            n_features = len(X_train.columns)

            # Define the parameter grid
            param_grid = {
                'max_iter': [int(min(MAX_ITER, int(0.1 * n_samples) - i * int(0.1 * n_samples) / 10))],
                'learning_rate': [LEARNING_RATE, 0.05, 0.01],
                'max_depth': [int(min(MAX_DEPTH, int(np.log2(n_features) - i * (np.log2(n_features) / 10))) if n_features > 0 else 10)],
                'min_samples_leaf': [int(max(MIN_SAMPLES_LEAF, int(0.01 * n_samples) - i * int(0.01 * n_samples) / 10))],
                'l2_regularization': [L2_REGULARIZATION],
                'validation_fraction': [min(VALIDATION_FRACTION, 1000 / n_samples - i / 10 * (1000 / n_samples))],
                'tol': [TOL]
            }

            # Initialize the model
            model = HistGradientBoostingClassifier(
                early_stopping=True,
                scoring="f1",
                class_weight='balanced'
            )

            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Best model from grid search
            best_model = grid_search.best_estimator_

            # Predict and evaluate performance for current fold
            y_test_pred = best_model.predict(X_test)
            dice.append(f1_score(y_test, y_test_pred, average='weighted'))

        # Update metrics after dimensionality reduction
        mean_test_dice = np.mean(dice)

        # Evaluate overall performance with reduced features
        y_pred_overall = best_model.predict(X)
        overall_dice = f1_score(y, y_pred_overall, average='weighted')

        print("\nMetrics after dimensionality reduction:")
        print(f"Mean test F1 score: {mean_test_dice:.2f}")
        print(f"Overall F1 score: {overall_dice:.2f}")
        i += 1

    return overall_dice, mean_test_dice, X, var_to_retain, X_scaled, best_model, mean_test_dice, overall_dice, pca


def common_dim_reduction(X,y):
    """
    Applies dimensionality reduction to a dataset to address overfitting.

    This function reduces the dimensionality of the dataset by dropping columns with a high percentage of missing values,
    imputing missing values, selecting features using Lasso, standardizing the features, and applying PCA to retain
    a specified amount of variance.

    Parameters:
    - X (DataFrame): Feature set.
    - y (Series): Target variable.

    Returns:
    - X_pca (array-like): Transformed feature set after PCA.
    - X (DataFrame): DataFrame of the transformed feature set.
    - y (Series): Target variable.
    - original_feature_names (list): Names of the original selected features.
    - var_to_retain (float): Variance retained after PCA.
    - X_scaled (array-like): Scaled feature set before PCA.
    """
    print("\nOverfitting detected. Applying dimensionality reduction...")

    # Drop columns with a high percentage of missing values
    na_threshold = 0.8
    na_counts = X.isnull().sum()
    na_columns = na_counts[na_counts / len(X) > na_threshold].index
    X = X.drop(columns=na_columns)

    # Impute missing values using a simple strategy
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Feature selection using Lasso
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_imputed, y)
    selected_features = X_imputed.columns[lasso.coef_ != 0]
    if len(selected_features) == 0:
        selected_features = X_imputed.columns
    X_selected = X_imputed[selected_features]
    original_feature_names = X_selected.columns

    # Standardize the selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Apply PCA to retain 80% of the variance
    var_to_retain = 0.8
    pca = PCA(n_components=var_to_retain)
    X_pca = pca.fit_transform(X_scaled)
    X = pd.DataFrame(X_pca, columns=[f'PCA_{i+1}' for i in range(X_pca.shape[1])])
    return X_pca, X, y, original_feature_names, var_to_retain, X_scaled

def random_forest(data, tar, tar_skew=True, pred_skew=True, columns_to_remove=None, 
                  identify_predictors=True, graphs=False, dim_reduce=True, categorical=False, n_splits = 5):
    """
    Train a regression or classification model using HistGradientBoostingRegressor/Classifier with cross-validation.
    
    Optionally applies skew correction, dimensionality reduction (Lasso + PCA) in case of
    overfitting, and displays performance metrics along with permutation-based feature importances.

    Parameters:
        data (pd.DataFrame): Input dataset.
        tar (str): Name of the target variable column.
        tar_skew (bool): Whether to correct skewness of the target variable.
        pred_skew (bool): Whether to correct skewness of predictor variables.
        columns_to_remove (list): List of columns to remove to avoid information leakage.
        identify_predictors (bool): If True, displays permutation-based feature importances.
        graphs (bool): If True, displays graphs comparing actual vs. predicted values.
        dim_reduce (bool): If True, applies dimensionality reduction (Lasso + PCA) when overfitting is detected.
        categorical (bool): If True, treats the target as categorical and uses a classifier.

    Returns:
        list: A list containing:
            - The relative difference in MAE between overall and cross-validation.
            - The overall R² score.
            - The mean test R² score from cross-validation.
    """
    pca = None
    original_feature_names = None
    MAX_ITER = int((data.shape[0]*data.shape[1])**0.5)
    
    if columns_to_remove is None:
        columns_to_remove = []
    
    data = data.dropna(axis=1, how='all')
    data = data.loc[:, ~data.columns.duplicated()]

    data = data.drop(columns=columns_to_remove, axis=1)
    data = pd.get_dummies(data, drop_first=True)
    
    # Apply mapping function
    data, mappings = map_non_numeric_columns(data)
    
    # Fill NaN values with column means
    data.fillna(data.mean(), inplace=True)
    
    # Output mappings
    print("Mappings:")
    for col, mapping in mappings.items():
        print(f"Column '{col}': {mapping}")

    # Correct skewness for target variable if required
    if tar_skew:
        data[tar] = correct_skewness(data, tar)

    # Correct skewness for predictor variables if required
    if pred_skew:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col not in [tar, "Log" + tar]:
                data[col] = correct_skewness(data, col)

    # Define features and target
    X = data.drop(tar, axis=1)
    X.fillna(X.mean(), inplace=True)
    y = data[tar]

    # Convert boolean columns to integers
    bool_cols = X.select_dtypes('bool').columns
    if not bool_cols.empty:
        X = X.astype({col: 'int' for col in bool_cols})

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True)

    mse_list = []
    mae_list = []
    r2_list = []

    if categorical:
        dice = []
    
    # Cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        n_samples = len(X_train)
        n_features = len(X_train.columns)

        # Set hyperparameters based on training data
        min_samples_leaf = max(MIN_SAMPLES_LEAF, int(0.01 * n_samples))
        validation_fraction = min(VALIDATION_FRACTION, 1000 / n_samples)
        max_iter = min(MAX_ITER, int(0.1 * n_samples))
        max_depth = min(MAX_DEPTH, int(np.log2(n_features))) if n_features > 0 else 10
        learning_rate = LEARNING_RATE if n_samples < 1000 else 0.05

        if not categorical:
            # Initialize and train the regression model
            model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=L2_REGULARIZATION,
                early_stopping=True,
                validation_fraction=validation_fraction,
                scoring="neg_mean_squared_error",
                tol=TOL
            )
            model.fit(X_train, y_train)

            # Predict and evaluate performance for current fold
            y_test_pred = model.predict(X_test)
            mse_list.append(mean_squared_error(y_test, y_test_pred))
            mae_list.append(mean_absolute_error(y_test, y_test_pred))
            r2_list.append(stats.pearsonr(y_test, y_test_pred)[0]**2)
            
        else:
            # Initialize and train the classification model
            model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=L2_REGULARIZATION,
                early_stopping=True,
                validation_fraction=validation_fraction,
                scoring="f1_weighted",
                tol=TOL
            )
            model.fit(X_train, y_train)

            # Predict and evaluate performance for current fold
            y_test_pred = model.predict(X_test)
            dice.append(f1_score(y_test, y_test_pred, average='weighted'))  # Use F1 score for categorical targets

    # Calculate mean performance across folds
    if not categorical:
        mean_test_mse = np.mean(mse_list)
        mean_test_mae = np.mean(mae_list)
        mean_test_r2 = np.mean(r2_list)

        # Evaluate performance on the entire dataset
        y_pred_overall = model.predict(X)
        overall_mse = mean_squared_error(y, y_pred_overall)
        overall_mae = mean_absolute_error(y, y_pred_overall)
        overall_r2 = r2_score(y, y_pred_overall)

        overfit_metric = (overall_mae - mean_test_mae) / mean_test_mae
    else:
        mean_test_dice = np.mean(dice)

        # Evaluate performance on the entire dataset
        y_pred_overall = model.predict(X)
        overall_dice = f1_score(y, y_pred_overall, average='weighted')

        # Calculate overfit metric for classification
        overfit_metric = (overall_dice - mean_test_dice) / mean_test_dice

    # If overfitting is detected, apply dimensionality reduction

    if abs(overfit_metric) > OVERFIT_THRESHOLD and dim_reduce:
        X_pca, X, y, original_feature_names, var_to_retain, X_scaled= common_dim_reduction(X,y)
        # Re-run cross-validation with reduced features
        mse_list, mae_list, r2_list = [], [], []
        if categorical:
            dice = []
        if not categorical:
            mean_test_mse, mean_test_mae, mean_test_r2, overall_mse, overall_mae, overall_r2, X, var_to_retain, X_scaled, model, pca = \
                    reduce_regression_complexity(overall_mae, mean_test_mae, overall_r2, mean_test_r2, mse_list, mae_list, r2_list, X, y, var_to_retain, X_scaled, kf, MAX_ITER)
        else:
            overall_dice, mean_test_dice, X, var_to_retain, X_scaled, model, mean_test_dice, overall_dice, pca = \
                    reduce_classification_complexity(overall_dice, mean_test_dice, dice, X, y, var_to_retain, X_scaled, kf, MAX_ITER)
    # Plot PCA components if applicable
    if pca is not None and original_feature_names is not None and graphs:
        plot_pca_components(pca, X_scaled, data, original_feature_names, tar)

    # Print overall performance metrics
    if not categorical:
        print(f"Mean Overall test MSE: {mean_test_mse:.2f}")
        print(f"Mean Overall test MAE: {mean_test_mae:.2f}")
        print(f"Mean Overall test R²: {mean_test_r2:.2f}")
        print(f"Overall MSE: {overall_mse:.2f}")
        print(f"Overall MAE: {overall_mae:.2f}")
        print(f"Overall R²: {overall_r2:.2f}")
    else:
        print(f"Mean Overall test F1 score: {mean_test_dice:.2f}")
        print(f"Overall F1 score: {overall_dice:.2f}")

    # Plot graphs if requested
    if graphs:
        if not categorical:
            # Plot regression results
            plot_regression_results(y_test, y_test_pred, tar, mean_test_mse, overall_mse, 
                                    mean_test_r2, overall_r2)
        else:
            # Plot classification results
            plot_classification_results(y_test, y_test_pred, model, X_test, tar, 
                                        overall_dice, mean_test_dice)

        # Plot feature importance if requested
        if identify_predictors:
            plot_feature_importance(model, X, y, X.columns, tar, categorical)

    # Return performance metrics
    if not categorical:
        return [abs((overall_mae - mean_test_mae) / mean_test_mae), overall_r2, mean_test_r2], model
    else:
        return [abs((overall_dice - mean_test_dice) / mean_test_dice), overall_dice, mean_test_dice], model
    
