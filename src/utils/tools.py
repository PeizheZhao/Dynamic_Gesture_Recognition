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