import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.inspection import permutation_importance


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_regression_results(y_test, y_test_pred, tar, mean_test_mse, overall_mse, mean_test_r2, overall_r2):
    """
    Plot visualization graphs for regression model results.
    
    Parameters:
        y_test (pd.Series): Actual target values from the test set.
        y_test_pred (np.array): Predicted target values for the test set.
        tar (str): Name of the target variable.
        mean_test_mse (float): Mean MSE from cross-validation.
        overall_mse (float): MSE on the entire dataset.
        mean_test_r2 (float): Mean R² from cross-validation.
        overall_r2 (float): R² on the entire dataset.
    """
    # Calculate correlation metrics
    pearson_corr, _ = stats.pearsonr(y_test, y_test_pred)
    spearman_corr, p_value = stats.spearmanr(y_test, y_test_pred)
    
    # Print correlation metrics
    print("\nCorrelation metrics on test set:")
    print(f"R² (Pearson squared): {pearson_corr**2:.4f}")
    print(f"Spearman coefficient: {spearman_corr:.4f}")
    print(f"P-value (Spearman): {p_value:.4f}")
    
    # Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_pred, y_test, alpha=0.3, label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'k--', lw=2, label='Perfect Fit')
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.title(f"Actual vs Predicted Values ({tar})\nSpearman: {spearman_corr:.2f} Pearson: {pearson_corr:.2f}")
    plt.legend()
    plt.grid()
    plt.show()

    # Rank correlation plot
    plt.figure(figsize=(10, 6))
    plt.scatter(stats.rankdata(y_test_pred), stats.rankdata(y_test), alpha=0.3)
    plt.plot(stats.rankdata(y_test), stats.rankdata(y_test), 'k--', lw=2, label='Perfect Fit')
    plt.ylabel("Actual Ranks")
    plt.xlabel("Predicted Ranks")
    plt.title(f"{tar} Spearman Correlation: {spearman_corr:.2f}")
    plt.legend()
    plt.grid()
    plt.show()

    # Training vs Testing performance comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(['Training MSE', 'Testing MSE'], [overall_mse, mean_test_mse], color='lightblue')
    ax[0].set_ylabel('MSE')
    ax[0].set_title('Training vs Testing MSE')
    ax[0].grid(axis='y')

    ax[1].bar(['Training R²', 'Testing R²'], [overall_r2, mean_test_r2], color='lightblue')
    ax[1].set_ylabel('R²')
    ax[1].set_title('Training vs Testing R²')
    ax[1].grid(axis='y')

    plt.tight_layout()
    plt.show()

def plot_classification_results(y_test, y_test_pred, model, X_test, tar, overall_dice, mean_test_dice):
    """
    Plot visualization graphs for classification model results.
    
    Parameters:
        y_test (pd.Series): Actual target values from the test set.
        y_test_pred (np.array): Predicted target values for the test set.
        model: Trained classifier model.
        X_test (pd.DataFrame): Test feature data.
        tar (str): Name of the target variable.
        overall_dice (float): Overall F1 score on the entire dataset.
        mean_test_dice (float): Mean F1 score from cross-validation.
    """
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({tar})")
    plt.show()

    # ROC Curve (for binary classification only)
    if len(model.classes_) == 2:  # Check if the target is binary
        y_test_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({tar})')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    # Bar plot for F1 score (training vs testing)
    plt.figure(figsize=(8, 6))
    plt.bar(['Training F1', 'Testing F1'], [overall_dice, mean_test_dice], color='lightblue')
    plt.ylabel('F1 Score')
    plt.title('Training vs Testing F1 Score')
    plt.grid(axis='y')
    plt.show()

def plot_feature_importance(model, X, y, feature_names, tar, categorical=False):
    """
    Plot feature importance graphs for the model.
    
    Parameters:
        model: Trained model.
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        feature_names (list): List of feature names.
        tar (str): Name of the target variable.
        categorical (bool): Whether the target is categorical.
    """
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, 
        scoring='neg_mean_squared_error' if not categorical else 'accuracy'
    )

    # Get importance scores
    importances = result.importances_mean
    
    # Sort features by importance (show top 10)
    n_features_to_show = min(10, len(feature_names))
    sorted_idx = np.argsort(importances)[-n_features_to_show:]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='skyblue')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title(f'Top {n_features_to_show} Feature Importances ({tar})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Optional: Class-specific permutation importance (for multi-class classification)
    if categorical and hasattr(model, 'classes_') and len(model.classes_) > 2:
        try:
            # Calculate permutation importance for each class
            class_importances = []
            for class_idx, class_name in enumerate(model.classes_):
                result_class = permutation_importance(
                    model, X, (y == class_name).astype(int), n_repeats=10, random_state=42, scoring='accuracy'
                )
                class_importances.append(result_class.importances_mean)

            # Aggregate class-specific importances
            class_importance_aggregated = np.mean(class_importances, axis=0)

            # Plot class-specific feature importance
            plt.figure(figsize=(12, 8))
            for i, class_name in enumerate(model.classes_):
                plt.barh(range(len(sorted_idx)), class_importances[i][sorted_idx], align='center', 
                         label=f'Class {class_name}', alpha=0.7)
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Permutation Importance')
            plt.title(f'Top {n_features_to_show} Feature Importances by Class ({tar})')
            plt.legend(loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Class-specific permutation importances could not be computed: {e}")



def plot_pca_components(pca, X_pca, data, original_feature_names, tar):
    """
    Plot PCA component loadings as a heatmap and PCA1 vs. PCA2 scatter plot.

    Parameters:
        pca: Fitted PCA object.
        X_pca: Transformed PCA data.
        original_feature_names (list): List of original feature names.
        tar: Target variable used for coloring the scatter plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 1. PCA Component Loadings as Heatmap ---
    sns.heatmap(
        pca.components_, 
        annot=True, 
        cmap='coolwarm', 
        yticklabels=[f"PCA_{i+1}" for i in range(pca.n_components_)], 
        xticklabels=original_feature_names,
        annot_kws={"size": 8},
        fmt=".2f",
        ax=axes[0]
    )
    axes[0].set_title("PCA Component Loadings", fontsize=12)
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)
    axes[0].tick_params(axis='y', labelsize=8)
    
    # --- 2. PCA1 vs. PCA2 Scatter Plot (if at least two components exist) ---
    tar = data[tar]
    if isinstance(tar, pd.Series) and not pd.api.types.is_numeric_dtype(tar):
        tar_numeric = pd.factorize(tar)[0]  # Convert categories to integer labels
    else:
        tar_numeric = tar  # If already numeric, use as is

    scatter = axes[1].scatter(
        X_pca[:, 0], X_pca[:, 1], c=tar_numeric, cmap='viridis', edgecolors='k', alpha=0.7
    )
    if X_pca.shape[1] >= 2:
        scatter = axes[1].scatter(
            X_pca[:, 0], X_pca[:, 1], c=tar, cmap='viridis', edgecolors='k', alpha=0.7
        )
        axes[1].set_xlabel("PCA 1")
        axes[1].set_ylabel("PCA 2")
        axes[1].set_title("PCA1 vs. PCA2 Scatter Plot", fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label("Target Variable")
    
    plt.tight_layout()
    plt.show()
