import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.local_utility import timer, set_device, set_seed
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DEVICE = set_device()


def train_model(model, train_set, optimizer=None):
    """
    Trains a given model on the provided training dataset using mini-batch gradient descent.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_set (torch.utils.data.Dataset): The dataset used for training.

        val_set (torch.utils.data.Dataset): The validation dataset used to assess the model's performance.

    Returns:
        torch.nn.Module: The trained model after all epochs.

    Notes:
        - Uses CrossEntropyLoss as the loss function.
        - Uses Adam optimizer with a learning rate from `config.LEARNING_RATE`.
        - Moves both model and data to the specified device (CPU/GPU).
        - Performs multiple epochs of training with gradient descent updates.
    """
    set_seed(seed_torch=True)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE)

    model.to(DEVICE)
    model.train()

    loss_stats = {
        'train': [],
        'val': []
    }

    for epoch in range(config.EPOCHS):
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def evaluate_model(model, val_set):
    """
    Evaluates the given model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained neural network model to be evaluated.
        val_set (torch.utils.data.Dataset): The validation dataset.

    Returns:
        Tuple[float, float]: A tuple containing:
            - avg_loss (float): The average loss over the validation dataset.
            - accuracy (float): The accuracy of the model on the validation dataset.

    Notes:
        - Uses CrossEntropyLoss as the evaluation loss function.
        - Moves both the model and data to the specified device (CPU/GPU).
        - Disables gradient calculations to improve efficiency.
    """
    correct, total_example, total_loss = 0, 0, 0

    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model = model.to(DEVICE)
    model = model.eval()

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)

            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()
            total_example += len(labels)
            total_loss += criterion(logits, labels).item()

    accuracy = correct / total_example
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy


def final_test_evaluation(model, test_set):
    """
    Evaluate a trained model on a given test dataset and compute performance metrics.

    This function evaluates a given model on a provided test dataset, computes key classification
    metrics, and visualizes the results using a confusion matrix and an ROC curve. It also stores
    the evaluation results for further analysis.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_set (torch.utils.data.Dataset): The dataset to evaluate the model on.

    Returns:
        list: A list containing a dictionary with evaluation metrics:
            - 'Model': Name of the model evaluated (MLP).
            - 'Accuracy': Classification accuracy.
            - 'Precision': Precision score.
            - 'Recall': Recall score.
            - 'F1-Score': F1-score.
            - 'ROC-AUC': Receiver Operating Characteristic Area Under Curve (ROC-AUC).
            - '3-Fold CV ROC-AUC': Placeholder for cross-validation score (NIL).

    Notes:
        - The function moves the model to evaluation mode.
        - Computes metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
        - Plots the confusion matrix and ROC curve.
        - Uses softmax to extract probabilities for the positive class.
        - Uses sklearn's classification_report to print a summary of classification performance.

    Example:
        >>> model = MyTrainedModel()
        >>> test_set = MyTestDataset()
        >>> results = final_test_evaluation(model, test_set)
    """
    model = model.to(DEVICE)
    model = model.eval()

    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False)

    # implement moving to gpu/cpu across board

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features)

            # Get probability of class 1
            probs = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Metrics
    accu = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    # Print Metrics
    print("")
    print(
        f"Accuracy: {accu:.2f} | Recall: {recall:.2f} | Precision: {precision:.2f} | ROC-AUC: {roc_auc:.2f}")

    print("\n", "_________"*11, "\n")

    # Plot Confusion Matrix

    cmap = sns.color_palette('Blues_r')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, cbar=False, cmap=cmap,
                ax=axes[0], annot_kws={"size": 15, "color": 'black'})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('Actual Label')
    axes[0].set_title(f'Confusion Matrix', fontsize=15)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')

    axes[1].plot(fpr, tpr, lw=2, marker='.',
                 label=f"ROC curve area", color='blue')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC curve', fontsize=15)
    axes[1].annotate(f'AUC ={round(roc_auc, 2)}', xy=(0.7, 0.5), fontsize=15,)
    axes[1].legend()
    plt.suptitle("Decentralized Model", fontsize=22)
    plt.tight_layout(pad=1)
    plt.subplots_adjust()
    plt.show()

    print("_________"*11, '\n')
    print('Federated Learning Classification Report')
    print(classification_report(all_labels, all_preds))
    print("_________"*11)

    final_model = {'Accuracy': round(accu, 2),
                   'Precision': round(precision, 2),
                   'Recall': round(recall, 2),
                   'F1-Score': round(f1, 2),
                   'ROC-AUC': round(roc_auc, 2),
                   '3-Fold CV ROC-AUC ': "NIL"}

    return final_model
