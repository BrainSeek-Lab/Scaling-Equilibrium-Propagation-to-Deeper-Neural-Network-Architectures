import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F


class Resnet_13(nn.Module):
    """
    ResNet-13 architecture for CIFAR-10 classification.
    """
    
    def __init__(self, num_inputs=3, num_hiddens_1=128, num_hiddens_2=256, 
                 num_hiddens_3=512, num_hiddens_4=512, num_outputs=10):
        super(Resnet_13, self).__init__()
        
        self.num_hiddens_4 = num_hiddens_4
        self.layers = nn.ModuleDict()

        # ---- Block 1 ----
        self.layers["0_1"] = nn.Sequential(
            nn.Conv2d(num_inputs, num_hiddens_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hiddens_1)
        )
        self.layers["1_2"] = nn.Sequential(
            nn.Conv2d(num_hiddens_1, num_hiddens_1, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_hiddens_1)
        )
        self.layers["0_2"] = nn.Sequential(
            nn.Conv2d(num_inputs, num_hiddens_1, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_hiddens_1)
        )

        # ---- Block 2 ----
        self.layers["2_3"] = nn.Sequential(
            nn.Conv2d(num_hiddens_1, num_hiddens_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hiddens_2)
        )
        self.layers["2_4"] = nn.Sequential(
            nn.Conv2d(num_hiddens_1, num_hiddens_2, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_hiddens_2)
        )
        self.layers["3_4"] = nn.Sequential(
            nn.Conv2d(num_hiddens_2, num_hiddens_2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_hiddens_2)
        )

        # ---- Block 3 ----
        self.layers["4_5"] = nn.Sequential(
            nn.Conv2d(num_hiddens_2, num_hiddens_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hiddens_3)
        )
        self.layers["4_6"] = nn.Sequential(
            nn.Conv2d(num_hiddens_2, num_hiddens_3, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_hiddens_3)
        )
        self.layers["5_6"] = nn.Sequential(
            nn.Conv2d(num_hiddens_3, num_hiddens_3, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_hiddens_3)
        )

        # ---- Block 4 ----
        self.layers["6_7"] = nn.Sequential(
            nn.Conv2d(num_hiddens_3, num_hiddens_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hiddens_4)
        )
        self.layers["6_8"] = nn.Sequential(
            nn.Conv2d(num_hiddens_3, num_hiddens_4, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_hiddens_4)
        )
        self.layers["7_8"] = nn.Sequential(
            nn.Conv2d(num_hiddens_4, num_hiddens_4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_hiddens_4)
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.layers["8_9"] = nn.Linear(num_hiddens_4*2*2, num_outputs)

        # Non-linearity
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        nodes = {}

        # Node 0
        nodes[0] = x

        # Block 1
        nodes[1] = self.activation(self.layers["0_1"](nodes[0]))
        nodes[2] = self.activation(
            self.layers["1_2"](nodes[1]) + self.layers["0_2"](nodes[0])
        )

        # Block 2
        nodes[3] = self.activation(self.layers["2_3"](nodes[2]))
        nodes[4] = self.activation(
            self.layers["2_4"](nodes[2]) + self.layers["3_4"](nodes[3])
        )

        # Block 3
        nodes[5] = self.activation(self.layers["4_5"](nodes[4]))
        nodes[6] = self.activation(
            self.layers["4_6"](nodes[4]) + self.layers["5_6"](nodes[5])
        )

        # Block 4
        nodes[7] = self.activation(self.layers["6_7"](nodes[6]))
        nodes[8] = self.activation(
            self.layers["6_8"](nodes[6]) + self.layers["7_8"](nodes[7])
        )

        # Global average pooling + linear head
        # pooled = self.global_avg_pool(nodes[8])
        flattened = torch.flatten(nodes[8], start_dim=1)
        output = self.layers["8_9"](flattened)

        return output


class CIFAR10Trainer:
    """Trainer class for CIFAR-10 classification with Resnet13."""
    
    def __init__(self, batch_size=128, learning_rate=1e-3, epochs=20):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0.0
        
        print(f"Using device: {self.device}")
        
        # Initialize data loaders
        self._setup_data()
        
        # Initialize model, loss function, and optimizer
        self._setup_model()
    
    def _setup_data(self):
        """Setup CIFAR-10 data loaders with appropriate transforms."""
        # Training transforms with augmentation
        transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandAugment(num_ops=3, magnitude=9),
        transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            ),
        ])
        
        # Test transforms without augmentation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            ),
        ])
        
        # Create datasets
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def _setup_model(self):
        """Initialize model, loss function, and optimizer."""
        self.model = Resnet_13(
            num_inputs=3,
            num_hiddens_1=128,
            num_hiddens_2=256,
            num_hiddens_3=512,
            num_hiddens_4=512,
            num_outputs=10
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=2.5e-4)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total_samples += targets.size(0)
            correct_predictions += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(dim=1)
                
                total_samples += targets.size(0)
                correct_predictions += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct_predictions / total_samples
        return accuracy
    
    def save_model(self, filepath="Resnet13_cifar10wbn.pth"):
        """Save the model state dict."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def train(self):
        """Complete training loop."""
        print(f"\nStarting training for {self.epochs} epochs...")
        print("-" * 60)
        
        for epoch in range(1, self.epochs + 1):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate on test set
            test_acc = self.evaluate()
            
            # Print epoch results
            print(f"Epoch {epoch:2d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")
            
            # Save best model
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                # Uncomment to save the best model
                self.save_model()
                print(f" New best accuracy: {test_acc:.2f}%")
            
            print("-" * 60)
        
        print(f"\n Training completed!")
        print(f"Best test accuracy: {self.best_accuracy:.2f}%")


def main():
    """Main function to run the training."""
    # Configuration
    config = {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 300
    }
    
    print("Resnet13 Network - CIFAR-10 Classification")
    print("=" * 50)
    print(f"Configuration: {config}")
    print("=" * 50)
    
    # Initialize trainer and start training
    trainer = CIFAR10Trainer(**config)
    trainer.train()


if __name__ == "__main__":
    main()