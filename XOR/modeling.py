import torch
from torch import nn



class XORClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        self.out_activation = nn.Sigmoid()
    
    def forward(self, X):
        out = self.layers(X)
        out = self.out_activation(out)

        return out # logits

    
    def predict(self, X):
        out = self.layers(X)
        out = self.out_activation(out)
        out = torch.round(out[0])
        return int(out)

    def batch_predict(self, X):
        out = self.layers(X)
        out = self.out_activation(out)
        out = torch.round(out[0])
        return int(out)
    
