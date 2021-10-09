import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(75, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(75, 512)
        self.fc4 = nn.Linear(1024, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x1, x2):
        # MLP1
        out1 = self.fc1(x1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.relu(out1)
        z1 = out1
        # MLP2
        out2 = self.fc3(x2)
        out2 = self.relu(out2)
        z2 = out2

        out1_cat_out2 = torch.cat((out1, out2), dim=1)
        out1_cat_out2 = self.fc4(out1_cat_out2)
        z = out1_cat_out2
        out = self.classifier(z)
        return z1, z2, out

