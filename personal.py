import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Targets
data = {
    "DOG" : torch.tensor([1, 0, 0, 0], dtype=torch.uint8, device=device),
    "CAT" : torch.tensor([0, 1, 0, 0], dtype=torch.uint8, device=device),
    "BIRD":  torch.tensor([0, 0, 1, 0], dtype=torch.uint8, device=device),
    "FISH":  torch.tensor([0, 0, 0, 1], dtype=torch.uint8, device=device),
}
targets = {
    "DOG": 1,
    "CAT": 2/3,
    "BIRD": 1/3,
    "FISH": 0
}

# Neural Network

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(4, 1)
    def forward(self, x):
        x = x / x.sum()
        x = self.fc(x)
        return x

# Init model
model = NeuralNetwork()
model.fc.weight.data = torch.ones_like(model.fc.weight.data)
model.fc.bias.data = torch.zeros_like(model.fc.bias.data)
model.zero_grad()
if (torch.cuda.is_available()):
    model = model.cuda()

print("="*5, "BEFORE", "="*5)
for name, param in model.named_parameters():
    print(f'{name}: {param.data}')    

print("="*5, "TRAIN", "="*5)
# Optimizing Model
learning_rate = 0.01
iteration = 1000
for i in tqdm(range(iteration)):
    for key, value in data.items():
        model.zero_grad()
        output = model(value)
        target = torch.full_like(output, targets[key], device=device)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()

        for f in model.parameters():
            f.data.sub_(f.grad.data * learning_rate)

print("="*5, "AFTER", "="*5)
for name, param in model.named_parameters():
    print(f'{name}: {param.data}')    

print("="*5, "CHECK", "="*5)
for key, value in data.items():
    output = model(value)
    print(f'{key}: {round(output.cpu().detach().numpy()[0], 2)}, target: {targets[key]}')
