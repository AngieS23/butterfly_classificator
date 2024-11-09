import torch
import torch.nn as nn

from convnet import ConvNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

def train_model(train_loader, val_loader):
    num_classes = 4
    learning_rate = 0.001
    num_epochs = 10

    model = ConvNet(in_channels=3, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_epoch(train_loader, val_loader, model, criterion, optimizer)
        
    return model
        

def train_epoch(train_loader, val_loader, model, criterion, optimizer):
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()
    
    check_accuracy(train_loader, model, 'Train')
    check_accuracy(val_loader, model, 'Validation')


def check_accuracy(loader, model, mode):
    num_correct = 0
    num_samples = 0
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"{mode} Accuracy: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    
    model.train()  # Set the model back to training mode
    
def evaluate_model(model, train_loader, test_loader):
    check_accuracy(train_loader, model, "Train")
    check_accuracy(test_loader, model, "Test")

    # TODO(us): Add other metrics
    # TODO(us): Add confussion matrix

def main():
    train_dataset = datasets.ImageFolder(f'../split_data/train', transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(f'../split_data/test', transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder(f'../split_data/val', transform=transforms.ToTensor())

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = train_model(train_loader=train_loader, val_loader=val_loader)
    evaluate_model(model=model, train_loader=train_loader, test_loader=test_loader)

if __name__ == '__main__':
    main()