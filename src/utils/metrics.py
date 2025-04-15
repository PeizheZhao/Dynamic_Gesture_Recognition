import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calculate_precision(outputs, targets):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  precision_score(targets.view(-1), pred.view(-1), average = 'macro')


def calculate_recall(outputs, targets):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    return  recall_score(targets.view(-1), pred.view(-1), average = 'macro')

def calculate_class_accuracy(outputs, targets, num_classes):
    """
    Computes the accuracy for each class.
    
    Args:
    - outputs (Tensor): Model outputs.
    - targets (Tensor): Ground truth labels.
    - num_classes (int): Number of classes.
    
    Returns:
    - class_accuracy (dict): Accuracy for each class.
    """
    # Get the predicted classes
    _, pred = outputs.topk(1, 1, True)
    pred = pred.view(-1)

    # Initialize a dictionary to hold correct counts and total counts for each class
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    # Iterate over targets and predictions
    for target, prediction in zip(targets.view(-1), pred):
        total_counts[target.item()] += 1
        if target.item() == prediction.item():
            correct_counts[target.item()] += 1

    # Calculate accuracy for each class
    class_accuracy = [0.0] * num_classes
    for cls in range(num_classes):
        if total_counts[cls] > 0:
            class_accuracy[cls] = correct_counts[cls] / total_counts[cls] * 100.0
        else:
            class_accuracy[cls] = 0.0  # No samples for this class

    return class_accuracy