import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import load_mnist_data
from model import build_cnn_classifier

from attack import FGSM

def train_model(arch='A', model_path=None, epochs=5, batch_size=256, adv_train=False, epsilon=0.3):
    """
    Trains a model with optional adversarial training.
    """
    train_dir = os.path.abspath("MNIST/Training")
    test_dir = os.path.abspath("MNIST/Testing")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Model {arch} (Adv Train: {adv_train}) on {device}")

    train_loader, test_loader = load_mnist_data(train_dir, test_dir, batch_size=batch_size)
    model = build_cnn_classifier(arch=arch).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    if adv_train:
        attacker = FGSM(model, epsilon=epsilon)

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            if adv_train:
                # 50/50 mix of clean and adversarial examples
                model.eval()
                adv_images = attacker.attack(images, labels)
                model.train()
                images = torch.cat([images, adv_images], dim=0)
                labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Model {arch} Accuracy: {accuracy*100:.2f}%")

    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    # 1. Train Model A (Base Victim)
    train_model(arch='A', model_path="model_A.pth", epochs=5)
    
    # 2. Train Model B (For Transferability)
    train_model(arch='B', model_path="model_B.pth", epochs=5)
    
    # 3. Train Robust Model A (Adversarial Training)
    train_model(arch='A', model_path="model_A_robust.pth", epochs=5, adv_train=True)
