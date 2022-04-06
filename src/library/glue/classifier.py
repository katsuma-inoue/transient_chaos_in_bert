from torch import nn
import torch


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_size*2, num_labels)
        # self.activate = nn.Sigmoid()
        module_list = [self.linear1
                       ]  #  self.relu, self.linear2, self.activate]
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x
