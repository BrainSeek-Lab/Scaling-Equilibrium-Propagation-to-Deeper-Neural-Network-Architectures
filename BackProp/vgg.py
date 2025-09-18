import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class VGG7Hopfield(nn.Module): #using vgg7 as it was mentioned in other papers whereas in the paper it was considered vgg5 as 5 trainable layer 
    """
    Spatial/channel schedule:
      32²×3 → 16²×128 → 8²×256 → 4²×512 → 2²×512 → Dense
    """
    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: 32×32 → 16×16
            conv_bn_relu(3, 128),
            nn.MaxPool2d(2),

            # Stage 2: 16×16 → 8×8
            conv_bn_relu(128, 256),
            nn.MaxPool2d(2),

            # Stage 3: 8×8 → 4×4
            conv_bn_relu(256, 512),
            nn.MaxPool2d(2),

            # Stage 4: 4×4 → 2×2
            conv_bn_relu(512, 512),
            nn.MaxPool2d(2),
        )

        # After 4 pools → feature map = (512, 2, 2) → flatten = 2048
        self.classifier = nn.Linear(512 * 2 * 2, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)            # (B, 512, 2, 2)
        x = torch.flatten(x, 1)         # (B, 2048)
        return self.classifier(x)       # (B, num_classes)

def vgg7_cifar10(num_classes=10):
    return VGG7Hopfield(num_classes)

def vgg7_cifar100(num_classes=100):
    return VGG7Hopfield(num_classes)

# ─────────────────────────────────────────────
# CIFAR10 Loaders
# ─────────────────────────────────────────────
def get_cifar10_loaders(root="./data", bs=128, workers=4):
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root, train=True,  download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_set, bs, shuffle=True,  num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader
def get_cifar100_loaders(root="./data", bs=128, workers=4):
    # CIFAR-100 dataset statistics
    mean_100 = (0.5071, 0.4867, 0.4408)
    std_100  = (0.2675, 0.2565, 0.2761)

    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean_100, std_100),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_100, std_100),
    ])

    train_set = datasets.CIFAR100(root, train=True,  download=True, transform=train_tfms)
    test_set  = datasets.CIFAR100(root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_set, bs, shuffle=True,  num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  bs, shuffle=False, num_workers=workers, pin_memory=True)
    
    return train_loader, test_loader
# ─────────────────────────────────────────────
# Train/Eval loops
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, opt, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return loss_sum / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return loss_sum / total, 100.0 * correct / total

# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def main():
    epochs, bs = 300, 128
    lr, wd = 0.1, 5e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # tr_loader, te_loader = get_cifar100_loaders(bs=bs)
    tr_loader, te_loader = get_cifar10_loaders(bs=bs)

    # model = vgg7_cifar100().to(device)
    model = vgg7_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                    weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, opt, device)
        te_loss, te_acc = evaluate(model, te_loader, criterion, device)
        sched.step()

        print(f"Epoch {ep:03d}/{epochs} | train {tr_acc:5.2f}% | test {te_acc:5.2f}%")
        best = max(best, te_acc)

    print(f"Best test acc: {best:.2f}%")

if __name__ == "__main__":
    main()
