import torch
import torch.nn as nn

from convnet import ConvNet
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import AdamW
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torcheval.metrics.functional import multiclass_confusion_matrix
from torchvision import datasets, transforms
from csv_writer import write_confusion_matrix, write_epoch_metrics, write_final_results

def append_epoch_metrics(results, mode, epoch, metrics):
    results.append({'mode': mode, 'epoch': epoch, 'metric': 'loss', 'result': metrics[0]})
    results.append({'mode': mode, 'epoch': epoch, 'metric': 'accuracy', 'result': metrics[1]})
    results.append({'mode': mode, 'epoch': epoch, 'metric': 'f1', 'result': metrics[2]})
    results.append({'mode': mode, 'epoch': epoch, 'metric': 'precision', 'result': metrics[3]})
    results.append({'mode': mode, 'epoch': epoch, 'metric': 'recall', 'result': metrics[4]})

def save_test_metrics(metrics):
    results = []
    results.append({'metric': 'loss', 'result': metrics[0]})
    results.append({'metric': 'accuracy', 'result': metrics[1]})
    results.append({'metric': 'f1', 'result': metrics[2]})
    results.append({'metric': 'precision', 'result': metrics[3]})
    results.append({'metric': 'recall', 'result': metrics[4]})
    return results

def train_model(train_loader, val_loader, device):
    num_classes = 4
    learning_rate = 0.0005
    num_epochs = 20

    model = ConvNet(in_channels=3, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    results = []

    for epoch in range(num_epochs):
        print(f"\n\nEpoch [{epoch + 1}/{num_epochs}]")
        train_metrics , val_metrics = train_epoch(train_loader, val_loader, model, criterion, optimizer, device)
        append_epoch_metrics(results, 'training', epoch, train_metrics)
        append_epoch_metrics(results, 'validation', epoch, val_metrics)
        
    write_epoch_metrics(results)

    return model, criterion

def train_epoch(train_loader, val_loader, model, criterion, optimizer, device):
    train_accuracy_metric = MulticlassAccuracy(num_classes=4, average='macro')
    train_f1_metric = MulticlassF1Score(num_classes=4, average='macro')
    train_precision_metric = MulticlassPrecision(num_classes=4, average='macro')
    train_recall_metric = MulticlassRecall(num_classes=4, average='macro')

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

    print(f"Training Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train F1 Score: {train_f1_score:.4f}")
    print(f"Train Precision: {train_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f}")
    print('\n')
    train_metrics = [avg_train_loss, train_accuracy, train_f1_score, train_precision, train_recall]
    val_metrics = validate(model, val_loader, criterion, device, 'Validation')

    return train_metrics, val_metrics
    

def validate(model, loader, criterion, device, mode):
    model.eval()
    accuracy_metric = MulticlassAccuracy(num_classes=4, average='macro')
    f1_metric = MulticlassF1Score(num_classes=4, average='macro')
    precision_metric = MulticlassPrecision(num_classes=4, average='macro')
    recall_metric = MulticlassRecall(num_classes=4, average='macro')

    epoch_loss = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)
            epoch_loss += loss.item()

            # Update metrics for validation
            predictions = scores.argmax(dim=1)
            accuracy_metric.update(predictions, targets)
            f1_metric.update(predictions, targets)
            precision_metric.update(predictions, targets)
            recall_metric.update(predictions, targets)

    # Compute average validation metrics for the epoch
    avg_loss = epoch_loss / len(loader)
    accuracy = accuracy_metric.compute().item()
    f1_score = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()

    accuracy_metric.reset()
    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()

    print(f"{mode} Loss: {avg_loss:.4f}")
    print(f"{mode} Accuracy: {accuracy:.4f}")
    print(f"{mode} F1 Score: {f1_score:.4f}")
    print(f"{mode} Precision: {precision:.4f}")
    print(f"{mode} Recall: {recall:.4f}")

    return [avg_loss, accuracy, f1_score, precision, recall]

def calculate_confusion_matrix(model, data_loader, num_classes, device, mode):
    all_predictions = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            predictions = scores.argmax(dim=1)
            all_predictions.append(predictions)
            all_targets.append(targets)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    confusion_matrix = multiclass_confusion_matrix(
        all_predictions, all_targets, num_classes=num_classes
    )

    matrix = confusion_matrix.cpu().numpy()
    write_confusion_matrix(mode, matrix)

    print(f"{mode} Confusion Matrix:")
    print(confusion_matrix.cpu().numpy())


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    train_dataset = datasets.ImageFolder(f'../balanced_data/train', transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(f'../balanced_data/test', transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder(f'../balanced_data/val', transform=transforms.ToTensor())

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model , criterion = train_model(train_loader=train_loader, val_loader=val_loader, device=device)
    print('\n')

    test_results = validate(model, test_loader, criterion, device, 'Testing')
    results = save_test_metrics(test_results)
    write_final_results(results)

    print('\n')
    train_matrix = calculate_confusion_matrix(model, train_loader, 4, device, 'Training')
    test_matrix = calculate_confusion_matrix(model, val_loader, 4, device, 'Testing')


if __name__ == '__main__':
    main()