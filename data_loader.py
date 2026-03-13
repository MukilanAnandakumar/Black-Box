import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_mnist_data(train_dir, test_dir, batch_size=128):
    """
    Loads MNIST images from the local Training/Testing directory structure using PyTorch.
    Normalizes images to [0, 1] and reshapes to (1, 28, 28).
    """
    # Define transformations: Convert to grayscale (if needed), resize, to tensor, normalize
    # MNIST images are 28x28, grayscale. 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(), # This normalizes to [0, 1]
    ])

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_dir = os.path.abspath("MNIST/Training")
    test_dir = os.path.abspath("MNIST/Testing")
    train_loader, test_loader = load_mnist_data(train_dir, test_dir)
    print(f"Loaded {len(train_loader.dataset)} training images.")
    print(f"Loaded {len(test_loader.dataset)} testing images.")
    
    # Check a sample
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}") # Should be (128, 1, 28, 28)
    print(f"Label batch shape: {labels.shape}")
