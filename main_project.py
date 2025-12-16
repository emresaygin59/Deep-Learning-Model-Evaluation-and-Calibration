import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, densenet121
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
RESULTS_DIR = "proje_sonuclari"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print(f"Kullanılan Cihaz: {DEVICE}")


# ------------------------------------------------
# 1. Dataset Preparation
# ------------------------------------------------
def get_dataloaders(dataset_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10

    elif dataset_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100

    # Split training set into Train (90%) and Validation (10%)
    from torch.utils.data import random_split
    val_size = int(0.1 * len(trainset))
    train_size = len(trainset) - val_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader, num_classes


# ------------------------------------------------
# 2. Metric: Expected Calibration Error (ECE)
# ------------------------------------------------
class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()


# ------------------------------------------------
# 3. Calibration: Temperature Scaling
# ------------------------------------------------
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        return self.model(x) / self.temperature

    def set_temperature(self, valid_loader):
        # Move model and criteria to GPU if available
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Collect logits and labels from validation set
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                logits_list.append(self.model(x.cuda()))
                labels_list.append(y)

        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

        # Calculate metrics before scaling
        before_nll = nll_criterion(logits, labels).item()
        before_ece = ece_criterion(logits, labels)

        # Optimize temperature parameter using LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate metrics after scaling
        after_nll = nll_criterion(logits / self.temperature, labels).item()
        after_ece = ece_criterion(logits / self.temperature, labels)

        print(f"NLL: {before_nll:.4f} → {after_nll:.4f}")
        print(f"ECE: {before_ece:.4f} → {after_ece:.4f}")
        print(f"Best T: {self.temperature.item():.4f}")

        return before_ece, after_ece, self.temperature.item()


# ------------------------------------------------
# 3.5 Test NLL Calculation
# ------------------------------------------------
def compute_nll(model, loader, temp=1.0):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x) / temp
            loss = criterion(logits, y)
            total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------
# 4. Model Training
# ------------------------------------------------
def train_model(model_name, dataset_name, num_classes, trainloader, valloader):
    if model_name == 'ResNet18':
        model = resnet18(num_classes=num_classes)
    else:
        model = densenet121(num_classes=num_classes)

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_losses, val_losses, train_accs = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        correct, total, running_loss = 0, 0, 0

        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        train_losses.append(running_loss / len(trainloader))
        train_accs.append(100 * correct / total)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in valloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item()

        val_losses.append(val_loss / len(valloader))
        print(f"Epoch {epoch + 1}/{EPOCHS} completed")

    # Plot and save training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title("Train Accuracy")

    plt.savefig(f"{RESULTS_DIR}/{model_name}_{dataset_name}_training_curves.png")
    plt.close()

    return model


# ------------------------------------------------
# 5. Visualization: Reliability Diagrams
# ------------------------------------------------
def plot_reliability_diagram(model, loader, model_name, dataset_name, temp=1.0):
    model.eval()
    logits, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(DEVICE)) / temp)
            labels.append(y)

    logits = torch.cat(logits)
    labels = torch.cat(labels).to(DEVICE)

    softmaxes = torch.softmax(logits, 1)
    confs, preds = torch.max(softmaxes, 1)
    accs = preds.eq(labels)

    # Binning predictions by confidence
    bins = torch.linspace(0, 1, 16)
    bin_accs = []

    for l, u in zip(bins[:-1], bins[1:]):
        mask = confs.gt(l) * confs.le(u)
        bin_accs.append(accs[mask].float().mean().item() if mask.sum() > 0 else 0)

    # Plot diagram
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], '--')
    plt.bar(np.linspace(0.05, 0.95, 15), bin_accs, width=1 / 15)
    status = "Calibrated" if temp != 1.0 else "Uncalibrated"
    plt.title(f"{model_name} {dataset_name} ({status})")
    plt.savefig(f"{RESULTS_DIR}/{model_name}_{dataset_name}_{status}_reliability.png")
    plt.close()


# ------------------------------------------------
# 6. Main Execution Loop
# ------------------------------------------------
def main():
    experiments = [
        ('ResNet18', 'CIFAR10'),
        ('ResNet18', 'CIFAR100'),
        ('DenseNet121', 'CIFAR10'),
        ('DenseNet121', 'CIFAR100')
    ]

    log = open(f"{RESULTS_DIR}/benchmark_logs.txt", "w")
    log.write("MODEL,DATASET,ECE_BEFORE,ECE_AFTER,NLL_BEFORE,NLL_AFTER,T\n")

    for model_name, dataset_name in experiments:
        # 1. Load Data
        trainloader, valloader, testloader, num_classes = get_dataloaders(dataset_name)

        # 2. Train Model
        base_model = train_model(model_name, dataset_name, num_classes, trainloader, valloader)

        # 3. Evaluate Uncalibrated Model
        plot_reliability_diagram(base_model, testloader, model_name, dataset_name)

        # 4. Perform Temperature Scaling
        scaled_model = ModelWithTemperature(base_model)
        ece_b, ece_a, T = scaled_model.set_temperature(valloader)

        # 5. Evaluate Calibrated Model
        nll_b = compute_nll(base_model, testloader, 1.0)
        nll_a = compute_nll(base_model, testloader, T)

        plot_reliability_diagram(base_model, testloader, model_name, dataset_name, T)

        # 6. Log Results
        log.write(f"{model_name},{dataset_name},{ece_b:.4f},{ece_a:.4f},{nll_b:.4f},{nll_a:.4f},{T:.4f}\n")
        log.flush()

        # Clean GPU memory
        torch.cuda.empty_cache()

    log.close()
    print("Tüm deneyler tamamlandı.")


if __name__ == "__main__":
    main()