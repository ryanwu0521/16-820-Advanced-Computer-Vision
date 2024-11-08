import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import torch
import torch.nn.functional as F
from torch import optim, nn
from nn import *
import torchvision
import torchvision.transforms as transforms

# fix random seed
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # horizontal flip
    transforms.RandomRotation(20),      # rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load 102 flowers dataset
trainset = torchvision.datasets.ImageFolder(root='./oxford-flowers102/train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='./oxford-flowers102/test', transform=transform)

# Data loader
batch_size = 64

# Custom CNN model (3 conv layers, 2 fc layers)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512) # Adjust for the size after conv and pooling
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Adjust for the size after conv and pooling
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(): 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Model 1 [Custom CNN]
    CustomCNN_model = CustomCNN(num_classes=102).to(device)

    # Model 2 [squeezenet 1.1] 
    Squeezenet_model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in Squeezenet_model.parameters():
        param.requires_grad = False

    # Modify the last layer
    Squeezenet_model.classifier[1] = nn.Conv2d(512, 102, kernel_size=(1,1), stride=(1,1))
    Squeezenet_model = Squeezenet_model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizers
    CustomCNN_model_optimizer = optim.Adam(CustomCNN_model.parameters(), lr=0.001)
    Squeezenet_model_optimizer = optim.Adam(Squeezenet_model.parameters(), lr=0.001)

    # Learning rate scheduler
    CustomCNN_model_scheduler = optim.lr_scheduler.CyclicLR(CustomCNN_model_optimizer, base_lr=0.003, max_lr=0.03)
    Squeezenet_model_scheduler = optim.lr_scheduler.CyclicLR(Squeezenet_model_optimizer, base_lr=0.003, max_lr=0.03)

    # Train the model
    max_iters = 50
    CustomCNN_train_loss = []
    CustomCNN_valid_loss = []
    CustomCNN_train_acc = []
    CustomCNN_valid_acc = []

    Squeezenet_train_loss = []
    Squeezenet_valid_loss = []
    Squeezenet_train_acc = []
    Squeezenet_valid_acc = []

    for epoch in range(max_iters):
        # Custom CNN model training and validation loop
        CustomCNN_model.train()
        CustomCNN_running_loss = 0.0
        CustomCNN_correct = 0
        CustomCNN_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            CustomCNN_model_optimizer.zero_grad()
            outputs = CustomCNN_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            CustomCNN_model_optimizer.step()
            CustomCNN_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            CustomCNN_total += labels.size(0)
            CustomCNN_correct += (predicted == labels).sum().item()
        CustomCNN_train_loss.append(CustomCNN_running_loss / len(train_loader))
        CustomCNN_train_acc.append(100 * CustomCNN_correct / CustomCNN_total)

        # Validation
        CustomCNN_model.eval()
        CustomCNN_running_loss = 0.0
        CustomCNN_correct = 0
        CustomCNN_total = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = CustomCNN_model(inputs)
                loss = criterion(outputs, labels)
                CustomCNN_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                CustomCNN_total += labels.size(0)
                CustomCNN_correct += (predicted == labels).sum().item()
        CustomCNN_valid_loss.append(CustomCNN_running_loss / len(valid_loader))
        CustomCNN_valid_acc.append(100 * CustomCNN_correct / CustomCNN_total)


        # Squuezenet model training and validation loop
        Squeezenet_model.train()
        Squeezenet_running_loss = 0.0
        Squeezenet_correct = 0
        Squeezenet_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            Squeezenet_model_optimizer.zero_grad()
            outputs = Squeezenet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            Squeezenet_model_optimizer.step()
            Squeezenet_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            Squeezenet_total += labels.size(0)
            Squeezenet_correct += (predicted == labels).sum().item()

        Squeezenet_train_loss.append(Squeezenet_running_loss / len(train_loader))
        Squeezenet_train_acc.append(100 * Squeezenet_correct / Squeezenet_total)

        # Validation
        Squeezenet_model.eval()
        Squeezenet_running_loss = 0.0
        Squeezenet_correct = 0
        Squeezenet_total = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = Squeezenet_model(inputs)
                loss = criterion(outputs, labels)
                Squeezenet_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                Squeezenet_total += labels.size(0)
                Squeezenet_correct += (predicted == labels).sum().item()
        Squeezenet_valid_loss.append(Squeezenet_running_loss / len(valid_loader))
        Squeezenet_valid_acc.append(100 * Squeezenet_correct / Squeezenet_total)

        # Learning rate scheduler step
        CustomCNN_model_scheduler.step()
        Squeezenet_model_scheduler.step()

        # Print epoch results
        print("itr: {:02d} \t CustomCNN_loss: {:.4f} \t CustomCNN_acc: {:.4f} \t Squeezenet_loss: {:.4f} \t Squeezenet_acc: {:.4f}".format(epoch, CustomCNN_train_loss[-1], CustomCNN_train_acc[-1], Squeezenet_train_loss[-1], Squeezenet_train_acc[-1]))

    # plot loss curves
    plt.plot(range(len(CustomCNN_train_loss)), CustomCNN_train_loss, label="CustomCNN training")
    # plt.plot(range(len(CustomCNN_valid_loss)), CustomCNN_valid_loss, label="CustomCNN validation")
    plt.plot(range(len(Squeezenet_train_loss)), Squeezenet_train_loss, label="SqueezeNet training")
    # plt.plot(range(len(Squeezenet_valid_loss)), Squeezenet_valid_loss, label="SqueezeNet validation")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    # plt.xlim(0, len(Squeezenet_train_loss) - 1)
    plt.xlim(0, len(CustomCNN_train_loss) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()

    # plot accuracy curves
    plt.plot(range(len(CustomCNN_train_acc)), CustomCNN_train_acc, label="CustomCNN training")
    # plt.plot(range(len(CustomCNN_valid_acc)), CustomCNN_valid_acc, label="CustomCNN validation")
    plt.plot(range(len(Squeezenet_train_acc)), Squeezenet_train_acc, label="SqueezeNet training")
    # plt.plot(range(len(Squeezenet_valid_acc)), Squeezenet_valid_acc, label="SqueezeNet validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.xlim(0, len(Squeezenet_train_acc) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()


