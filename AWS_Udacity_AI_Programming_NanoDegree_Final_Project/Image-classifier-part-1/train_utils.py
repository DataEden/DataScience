import os
import time
import torch
import torchvision
#s Scaling and mixed precision for better training and performance.
from torchvision import datasets, transforms, models 
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import VGG16_Weights
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Enable benchmark mode for faster GPU performance
torch.backends.cudnn.benchmark = True

# Load and preprocess the dataset
def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = valid_transforms

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define dataloaders with multiple workers for speed
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
        'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=32, num_workers=4, pin_memory=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)
    }

    return dataloaders, train_dataset.class_to_idx

# Initialize Model with Correct Weights
def initialize_model(arch, hidden_units, learning_rate, use_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    # Dictionary of supported models
    available_models = {
        "vgg16": models.vgg16,
        "resnet50": models.resnet50,
        "densenet121": models.densenet121,
        "mobilenet_v3_large": models.mobilenet_v3_large
    }

    # Ensure valid architectures
    valid_architectures = list(available_models.keys())

    # Check if selected architecture is valid
    if arch not in available_models:
        raise ValueError(
            f"Unsupported architecture: {arch}. Choose from: {', '.join(valid_architectures)}.\n"
            f"For example: '--arch vgg16' or '--arch resnet50'."
        )

    # Load selected pre-trained model
    model = available_models[arch](weights="DEFAULT")

    # Freeze feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Determine input features dynamically
    if hasattr(model, "classifier"):  # VGG, DenseNet, etc...
        if isinstance(model.classifier, nn.Sequential):  # VGG-type models
            input_features = model.classifier[0].in_features
        elif isinstance(model.classifier, nn.Linear):  # DenseNet-type models
            input_features = model.classifier.in_features
        else:
            raise ValueError(f"Unexpected classifier type for architecture: {arch}")
    elif hasattr(model, "fc"):  # ResNet, MobileNet, etc...
        input_features = model.fc.in_features
    else:
        raise ValueError(f"Unable to determine input features for the architecture: {arch}")

    # Replace classifier to match the dataset
    if hasattr(model, "classifier"):  # VGG, DenseNet
        print(f"Using `classifier` for {arch}")
        model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer_params = model.classifier.parameters()  # Optimizer updates classifier
    elif hasattr(model, "fc"):  # ResNet, MobileNet
        print(f"Using `fc` for {arch}")
        model.fc = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer_params = model.fc.parameters()  # Optimizer updates fc
    else:
        raise ValueError(f"Unable to replace classifier for the architecture: {arch}")

    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(optimizer_params, lr=learning_rate, weight_decay=1e-4, betas=(0.85, 0.999))

    # Learning rate schedulers
    step_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Mixed precision training
    scaler = GradScaler()

    # Ensure proper indentation of return statement
    return model, criterion, optimizer, scaler, step_scheduler, plateau_scheduler, device

# Training function for a single epoch
def train_one_epoch(model, dataloaders, criterion, optimizer, device):
    model.train()
    running_loss = 0
    scaler = GradScaler()  # Enable mixed precision training

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast():
            logps = model(inputs)
            loss = criterion(logps, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(dataloaders['train'])


# Validation function
def validate_model(model, dataloaders, criterion, device):
    model.eval()
    val_loss = 0
    accuracy = 0

    with torch.no_grad():
        for val_inputs, val_labels in dataloaders['valid']:
            val_inputs, val_labels = val_inputs.to(device, non_blocking=True), val_labels.to(device, non_blocking=True)

            logps = model(val_inputs)
            batch_loss = criterion(logps, val_labels)
            val_loss += batch_loss.item()

            # Accuracy calculation
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == val_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    val_loss /= len(dataloaders['valid'])
    accuracy /= len(dataloaders['valid'])
    return val_loss, accuracy


# Model training loop
def train_model(dataloaders, arch, hidden_units, learning_rate, epochs, use_gpu):
    model, criterion, optimizer, scaler, step_scheduler, plateau_scheduler, device = initialize_model(arch, hidden_units, learning_rate, use_gpu)
       
    best_val_loss = float('inf')
    early_stop_count = 0
    early_stop_threshold = 3  

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, dataloaders, criterion, optimizer, device)
        val_loss, accuracy = validate_model(model, dataloaders, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}")
        
        # Apply StepLR every 3 epochs
        if (epoch + 1) % 3 == 0:
            step_scheduler.step()
            print(f"StepLR applied: LR updated to {optimizer.param_groups[0]['lr']}")

        # Apply ReduceLROnPlateau when validation loss doesn't improve
        plateau_scheduler.step(val_loss)
        print(f"ReduceLROnPlateau checked: LR updated to {optimizer.param_groups[0]['lr']}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0  
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_threshold:
            print("Early stopping triggered. Training halted.")
            break

    return model


# Save trained model as a checkpoint
def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units=None, learning_rate=None, epochs=None):
    """
    Saves the trained model checkpoint for future use...

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        save_dir (str): Directory to save the checkpoint.
        class_to_idx (dict): Mapping of class labels to indices.
        arch (str): Model architecture used (e.g., 'vgg16', 'resnet50').
        hidden_units (int, optional): Number of hidden units in the classifier.
        learning_rate (float, optional): Learning rate used during training.
        epochs (int, optional): Number of training epochs.
    """
    # Ensure correct layer is saved
    classifier_layer = model.classifier if hasattr(model, "classifier") else model.fc
    
    # Define checkpoint structure
    checkpoint = {
        'arch': arch,  
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,  # Ensure mapping is included
        'classifier': classifier_layer # Handles both VGG16 (classifier) and ResNet (fc)     
    }
    
    # Key renaming for ResNet's fc layer...
    if hasattr(model, "fc"):
        checkpoint['fc'] = model.fc  # Save fc layer separately for ResNet

    # Add optional hyperparameters if one is provided
    if hidden_units is not None:
        checkpoint['hidden_units'] = hidden_units
    if learning_rate is not None:
        checkpoint['learning_rate'] = learning_rate
    if epochs is not None:
        checkpoint['epochs'] = epochs
        
    # Save checkpoint with a dynamic filename including architecture
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    print(f"Saving to: {os.path.abspath(save_dir)}")
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{arch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Log save information
    print(f"Saving model with architecture: {arch}")
    print(f"Hyperparameters: hidden_units={hidden_units}, learning_rate={learning_rate}, epochs={epochs}")
    print(f"Model checkpoint saved to {checkpoint_path}")

    