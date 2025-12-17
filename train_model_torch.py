import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- 1. Configuration ---
DATA_DIR = 'dataset'
IMG_SIZE = 64
BATCH_SIZE = 8
# CRITICAL FOR PHASE 2: Increased EPOCHS for better generalization
EPOCHS = 150 
MODEL_PATH = 'ball_classifier.pth'

# --- 2. Model Definition ---
class BallClassifier(nn.Module):
    """
    A simple CNN for binary classification of billiard balls (solid vs. striped).
    Matches the architecture used in the detection phase (project3.py).
    """
    def __init__(self):
        super(BallClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Flattened size calculation: (64 / 2 / 2 / 2) = 8. Result: 64 * 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- Helper Function for Data Loading ---
def is_valid_image_file(path):
    """
    Filters out system files and hidden directories while ensuring 
    the file is an image.
    """
    return (
        os.path.isfile(path) and 
        not os.path.basename(path).startswith('.') and
        path.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'))
    )

# --- 3. Data Loading and Augmentation ---
if __name__ == '__main__':
    print("Preparing data...")
    
    # Define transformations for training
    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE), 
        # AGGRESSIVE ROTATION: Vital for handling rotated test cases in Phase 2
        transforms.RandomRotation(30),
        # COLOR JITTER: Moderate levels to help model handle lighting changes
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), 
        transforms.ToTensor(), 
    ])

    # Load dataset using ImageFolder
    image_datasets = datasets.ImageFolder(
        DATA_DIR, 
        data_transforms,
        is_valid_file=is_valid_image_file
    )

    # Split into 80% training and 20% validation
    train_size = int(0.8 * len(image_datasets))
    val_size = len(image_datasets) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(image_datasets, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    # --- 4. Training Loop ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BallClassifier().to(device)

    # Use Binary Cross Entropy Loss for binary classification
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting model training on {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        # Validation check after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Validation Acc: {epoch_acc:.4f}')

    # --- 5. Save the Model ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining complete. Model weights saved to '{MODEL_PATH}'")