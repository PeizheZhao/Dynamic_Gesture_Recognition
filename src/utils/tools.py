import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def memory():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2) 
        print(f"Allocated memory: {allocated_memory:.2f} MB")
        print(f"Reserved memory: {reserved_memory:.2f} MB")

class LabelLogger:
    def __init__(self):
        # Initialize empty tensors to store the outputs and true labels
        self.outputs_labels = None
        self.true_labels = None

    def log(self, outputs, labels):
        """
        Add a round of outputs and true labels to the log by stacking tensors.
        
        :param outputs: A tensor of outputs for one epoch
        :param labels: A tensor of true labels for one epoch
        """
        # Convert inputs to tensors if they are not already
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        
        # Stack the tensors
        if self.outputs_labels is None:
            self.outputs_labels = outputs
            self.true_labels = labels
        else:
            self.outputs_labels = torch.cat((self.outputs_labels, outputs), dim=0)
            self.true_labels = torch.cat((self.true_labels, labels), dim=0)

    def clear(self):
        """Clear all logged data."""
        self.outputs_labels = None
        self.true_labels = None

    def get_logged_data(self):
        """
        Return all stored outputs and true labels.
        
        :return: (outputs_labels, true_labels)
        """
        return self.outputs_labels, self.true_labels