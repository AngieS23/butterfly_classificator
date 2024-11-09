import torch
import torch.nn as nn

from convnet import ConvNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchvision import datasets, transforms

def train_model(train_loader, val_loader, device):
    num_classes = 4
    learning_rate = 0.001
    num_epochs = 10

    model = ConvNet(in_channels=3, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_epoch(train_loader, val_loader, model, criterion, optimizer, device)
        
    return model

def train_epoch(train_loader, val_loader, model, criterion, optimizer, device):
    train_accuracy_metric = MulticlassAccuracy(num_classes=4)
    train_f1_metric = MulticlassF1Score(num_classes=4)
    train_precision_metric = MulticlassPrecision(num_classes=4)
    train_recall_metric = MulticlassRecall(num_classes=4)

    model.train()
    epoch_train_loss = 0 

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(data)
        loss = criterion(scores, targets)
        epoch_train_loss += loss.item()

        # Update metrics for training
        predictions = scores.argmax(dim=1)
        train_accuracy_metric.update(predictions, targets)
        train_f1_metric.update(predictions, targets)
        train_precision_metric.update(predictions, targets)
        train_recall_metric.update(predictions, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_accuracy = train_accuracy_metric.compute().item()
    train_f1_score = train_f1_metric.compute().item()
    train_precision = train_precision_metric.compute().item()
    train_recall = train_recall_metric.compute().item()

    train_accuracy_metric.reset()
    train_f1_metric.reset()
    train_precision_metric.reset()
    train_recall_metric.reset()

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1_score:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")

    validate_epoch(model, val_loader, criterion, device)

    

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_accuracy_metric = MulticlassAccuracy(num_classes=4)
    val_f1_metric = MulticlassF1Score(num_classes=4)
    val_precision_metric = MulticlassPrecision(num_classes=4)
    val_recall_metric = MulticlassRecall(num_classes=4)

    epoch_val_loss = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)
            epoch_val_loss += loss.item()

            # Update metrics for validation
            predictions = scores.argmax(dim=1)
            val_accuracy_metric.update(predictions, targets)
            val_f1_metric.update(predictions, targets)
            val_precision_metric.update(predictions, targets)
            val_recall_metric.update(predictions, targets)

    # Compute average validation metrics for the epoch
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_accuracy = val_accuracy_metric.compute().item()
    val_f1_score = val_f1_metric.compute().item()
    val_precision = val_precision_metric.compute().item()
    val_recall = val_recall_metric.compute().item()

    val_accuracy_metric.reset()
    val_f1_metric.reset()
    val_precision_metric.reset()
    val_recall_metric.reset()

    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1_score:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

    
def evaluate_model(model, train_loader, test_loader):
    # TODO(us): Implement
    # TODO(us): Add other metrics
    # TODO(us): Add confussion matrix
    pass

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    train_dataset = datasets.ImageFolder(f'../split_data/train', transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(f'../split_data/test', transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder(f'../split_data/val', transform=transforms.ToTensor())

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = train_model(train_loader=train_loader, val_loader=val_loader, device=device)
    evaluate_model(model=model, train_loader=train_loader, test_loader=test_loader)

if __name__ == '__main__':
    main()