import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method")

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

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
