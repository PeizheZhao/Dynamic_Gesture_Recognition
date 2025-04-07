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