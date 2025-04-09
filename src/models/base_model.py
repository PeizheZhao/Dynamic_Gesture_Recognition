import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def get_optimizer(self, lr, optimizer_type='adam', **kwargs):
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=lr, **kwargs)
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=lr, **kwargs)
        elif optimizer_type.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=lr, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Please choose from 'adam', 'sgd', 'rmsprop'.")

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def log_model_info(self):
        print("Model structure:")
        print(self)
        print("Model parameters:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.size()}")

    def calculate_parameter_size(self):
        total_memory_bytes = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_memory_mb = total_memory_bytes / (1024 ** 2)  
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        print(f"Total parameters: {total_memory_bytes}") 
        print(f"Total memory: {total_memory_mb:.2f} MB")
