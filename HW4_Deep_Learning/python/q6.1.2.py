import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import torch
import torch.nn.functional as F
from torch import optim, nn
from nn import *

# fix random seed
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# Extract data
train_x, train_y = train_data["train_data"], train_data["train_labels"]
valid_x, valid_y = valid_data["valid_data"], valid_data["valid_labels"]
test_x, test_y = test_data["test_data"], test_data["test_labels"]

# convert to pytorch tensors (CNN)
train_x = torch.tensor(train_x, dtype=torch.float32).view(-1, 1, 32, 32).to(device)
train_y = torch.tensor(np.argmax(train_y, axis=1), dtype=torch.long).to(device)
valid_x = torch.tensor(valid_x, dtype=torch.float32).view(-1, 1, 32, 32).to(device)
valid_y = torch.tensor(np.argmax(valid_y, axis=1), dtype=torch.long).to(device)
test_x = torch.tensor(test_x, dtype=torch.float32).view(-1, 1, 32, 32).to(device)
test_y = torch.tensor(np.argmax(test_y, axis=1), dtype=torch.long).to(device)

# Data loader
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Define model parameters
max_iters = 50
learning_rate = 0.003
hidden_size = 64
input_size = train_x.shape[1]
output_size = len(torch.unique(train_y))

# Define model (CNN)
class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Initialize model
model = CNN(output_size).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.003, max_lr=0.03)

# Training loop
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

for epoch in range(max_iters):
    model.train()
    total_loss = 0
    avg_acc = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_acc += (y_pred.argmax(dim=1) == yb).float().mean().item()

    train_loss.append(total_loss / len(train_loader))
    train_acc.append(avg_acc / len(train_loader))

    model.eval()
    total_loss = 0
    avg_acc = 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            total_loss += loss.item()
            avg_acc += (y_pred.argmax(dim=1) == yb).float().mean().item()
    
    valid_loss.append(total_loss / len(valid_loader))
    valid_acc.append(avg_acc / len(valid_loader))

    scheduler.step()
    
    print("itr: {:02d} \t loss: {:.4f} \t acc : {:.4f}".format(epoch, train_loss[-1], train_acc[-1]))


# plot loss curves
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(valid_loss)), valid_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss) - 1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()

# plot accuracy curves
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(valid_acc)), valid_acc, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc) - 1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()
    