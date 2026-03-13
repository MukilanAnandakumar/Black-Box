import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN_A(nn.Module):
    """
    Architecture A: Standard 2-layer CNN
    """
    def __init__(self, num_classes=10):
        super(MNIST_CNN_A, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNIST_CNN_B(nn.Module):
    """
    Architecture B: Deeper CNN with Dropout for transferability testing
    """
    def __init__(self, num_classes=10):
        super(MNIST_CNN_B, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def build_cnn_classifier(arch='A', num_classes=10):
    if arch == 'A':
        return MNIST_CNN_A(num_classes)
    elif arch == 'B':
        return MNIST_CNN_B(num_classes)
    else:
        raise ValueError("Unknown architecture type")

if __name__ == "__main__":
    model = build_cnn_classifier()
    print(model)
    # Test with sample input
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
