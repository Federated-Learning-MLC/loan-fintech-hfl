import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_model(model, train_set):

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)
    
    model = model.train()
    
    for epoch in range(config.EPOCHS):
        for features, labels in train_loader:
            logits = model(features)
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
def evaluate_model(model, val_set):
    correct, total_example, total_loss = 0, 0, 0
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model = model.eval()
    
    for features, labels in val_loader:
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels).item()
        total_example += len(labels)
        total_loss += criterion(logits, labels).item()
    
    accuracy = correct / total_example
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy


final_model = []

def final_test_evaluation(model, test_set):
    """_summary_

    Args:
        model (_type_): _description_
        test_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = model.eval()
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    #implement moving to gpu/cpu across board
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1] # Get probability of class 1
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
    print(f"Accuracy: {accu:.2f} | Recall: {recall:.2f} | Precision: {precision:.2f} | ROC-AUC: {roc_auc:.2f}")

    print("\n", "_________"*11, "\n")

    # Plot Confusion Matrix
    cmap = sns.color_palette('Blues_r') 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, cbar=False, cmap=cmap, ax=axes[0], annot_kws={"size": 15, "color": 'black'})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('Actual Label')
    axes[0].set_title(f'MLP Confusion Matrix', fontsize=15)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    axes[1].plot(fpr, tpr, lw=2, marker='.', label=f"MLP ROC curve area", color='blue')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('MLP ROC curve', fontsize=15)
    axes[1].annotate(f'AUC ={round(roc_auc, 2)}', xy=(0.7, 0.5), fontsize=15,)
    axes[1].legend()
    plt.suptitle("MLP Model", fontsize=22)
    plt.tight_layout(pad=1)
    plt.subplots_adjust()
    plt.show()

    print("_________"*11, '\n')
    print('Federated Learning Classification Report')
    print(classification_report(all_labels, all_preds))
    print("_________"*11)

    final_model.append({'Model': "MLP",
                        'Accuracy': round(accu, 2),
                        'Precision': round(precision, 2),
                        'Recall': round(recall, 2),
                        'F1-Score': round(f1, 2),
                        'ROC-AUC': round(roc_auc, 2),
                        '3-Fold CV ROC-AUC ': "NIL"})

    return final_model