import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.local_utility import timer

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Set preffered colormap
cmap = sns.color_palette('Blues_r')

# Define list to store result
final_model = list()


@timer
def train_predict(model, xtrain, ytrain, xtest, ytest, model_name=''):
    """
    Train a machine learning model and evaluate its performance.

    This function trains the given model on a training dataset, makes predictions 
    on both training and test datasets, and evaluates various classification metrics.
    Additionally, it performs 3-fold cross-validation and visualizes model performance 
    using a confusion matrix and ROC curve.

    Args:
        model (sklearn.base.BaseEstimator): The machine learning model to train.
        xtrain (np.ndarray or pd.DataFrame): Feature matrix for training.
        ytrain (np.ndarray or pd.Series): Target labels for training.
        xtest (np.ndarray or pd.DataFrame): Feature matrix for testing.
        ytest (np.ndarray or pd.Series): Target labels for testing.
        model_name (str, optional): Name of the model for visualization and reporting. 
                                    Defaults to an empty string.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries containing 
        model performance metrics, including:
            - Accuracy
            - Precision
            - Recall
            - F1-Score
            - ROC-AUC Score
            - 3-Fold Cross-Validation ROC-AUC Score

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> train_metrics = train_predict(model, x_train, y_train, x_test, y_test, model_name="RF")
        >>> print(train_metrics)
    """
    model.fit(xtrain, ytrain)
    y_preds_on_train = model.predict(xtrain)
    y_preds = model.predict(xtest)
    y_preds_proba = model.predict_proba(xtest)[:, 1]

    # 3-fold cross validation
    kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_score = cross_val_score(
        model, xtest, ytest, scoring='roc_auc', cv=kfolds).mean()

    roc_auc = roc_auc_score(ytest, y_preds_proba)
    recall = recall_score(ytest, y_preds)
    precision = precision_score(ytest, y_preds)
    accuracy_on_train = accuracy_score(ytrain, y_preds_on_train)
    accuracy_on_test = accuracy_score(ytest, y_preds)
    f1score = f1_score(ytest, y_preds)

    print("\n", "_________"*11, "\n")
    print(f"Train Accuracy: {accuracy_on_train:.2f} | Test Accuracy: {accuracy_on_test:.2f} |"
          f" Recall: {recall:.2f} | Precision: {precision:.2f} | ROC-AUC: {roc_auc:.2f}")
    print(f"\nROC-AUC After 3-Fold Cross Validation: {cv_score: .2f}")

    print("")

    cf = confusion_matrix(ytest, y_preds, normalize='true')

    # Plot Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cf, annot=True, cbar=False, cmap=cmap,
                ax=axes[0], annot_kws={"size": 15, "color": 'black'})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('Actual Label')
    axes[0].set_title(f'{model_name} Confusion Matrix', fontsize=15)

    # Plot ROC curve
    y_preds_prob = model.predict_proba(xtest)[:, 1]
    fpr, tpr, thresholds = roc_curve(ytest, y_preds_prob)

    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    axes[1].plot(fpr, tpr, lw=2, marker='.',
                 label=f"{model_name} ROC curve area", color='blue')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{model_name} ROC curve', fontsize=15)
    axes[1].annotate(f'AUC ={round(roc_auc, 2)} ', xy=(0.7, 0.5), fontsize=15,)
    axes[1].legend()
    plt.suptitle(model_name + ' Model', fontsize=22)
    plt.tight_layout(pad=1)
    plt.subplots_adjust()
    plt.show()

    print("_________"*11, '\n')
    print(f'{model_name} Classification Report')
    print(classification_report(ytest, y_preds))

    final_model.append({'Model': model_name,
                        'Accuracy': round(accuracy_on_test, 2),
                        'Precision': round(precision, 2),
                        'Recall': round(recall, 2),
                        'F1-Score': round(f1score, 2),
                        'ROC-AUC': round(roc_auc, 2),
                        '3-Fold CV ROC-AUC ': round(cv_score, 2)})

    return final_model


# -------------------------------- HYPERPARAMETER TUNNING ---------------------------------

def objective(X_train, y_train, model_name: str):
    """
    Defines the objective function for hyperparameter tuning of different machine learning models using Optuna.

    This function creates separate objective functions for:
    - Random Forest (`objective_rf`)
    - LightGBM (`objective_lgbm`)
    - XGBoost (`objective_xgb`)

    The function selects the appropriate objective function based on `model_name` and returns it for Optuna optimization.

    Args:
        X_train (pd.DataFrame or np.ndarray): The training feature matrix.
        y_train (pd.Series or np.ndarray): The training target labels.
        model_name (str): The name of the model to optimize.
                          Options: `"random forest"`, `"xgboost"`, `"lightgbm"`.

    Returns:
        Callable: The selected objective function for Optuna optimization.

    Example:
        >>> study = optuna.create_study(direction="maximize")
        >>> study.optimize(objective(X_train, y_train, "random forest"), n_trials=50)
    """

    def objective_rf(trial) -> float:
        """
        Objective function for tuning a RandomForestClassifier.

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

        Returns:
            float: Mean ROC-AUC score from cross-validation.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 100000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
        }

        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train,
                                cv=3, scoring="roc_auc").mean()

        return score

    def objective_lgbm(trial) -> float:
        """
        Objective function for tuning a LightGBMClassifier.

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

        Returns:
            float: Mean ROC-AUC score from cross-validation.
        """
        params = params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 100000, step=1000),
            "max_depth": trial.suggest_int("max_depth", 2, 13),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 4),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        }

        model = LGBMClassifier(**params, verbose=0, random_state=42)
        score = cross_val_score(model, X_train, y_train,
                                cv=3, scoring="roc_auc").mean()

        return score

    def objective_xgb(trial) -> float:
        """
        Objective function for tuning a XGBClassifier.

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

        Returns:
            float: Mean ROC-AUC score from cross-validation.
        """
        params = {
            'n_estimators': trial.suggest_int("n_estimators", 1000, 100000, step=1000),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 4),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0, step=0.2),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 5.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.2)
        }

        model = XGBClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train,
                                cv=3, scoring="roc_auc").mean()

        return score

    if model_name.lower() == "random forest":
        objective_function = objective_rf

    elif model_name.lower() == "xgboost":
        objective_function = objective_xgb

    else:
        objective_function = objective_lgbm

    return objective_function
