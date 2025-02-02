import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

# Load and preprocess the dataset
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

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

    # Define dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=32),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=32)
    }

    return dataloaders, train_dataset.class_to_idx

# Train the model
def train_model(dataloaders, arch, hidden_units, learning_rate, epochs, use_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    # Load a pre-trained model
    model = getattr(torchvision.models, arch)(pretrained=True)

    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Get input features dynamically
    input_features = model.classifier.in_features  

    # Replace classifier with a new one
    model.classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),  
        nn.LogSoftmax(dim=1)
    )

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scaler = GradScaler()  # Mixed precision training

    best_val_loss = float('inf')
    early_stop_count = 0
    early_stop_threshold = 3  # Stop if no improvement for 3 epochs

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():  # Mixed precision for speed
                logps = model(inputs)
                loss = criterion(logps, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, accuracy = 0, 0
        with torch.no_grad():
            for val_inputs, val_labels in dataloaders['valid']:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

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

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {accuracy:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0  # Reset counter
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_threshold:
            print("Early stopping triggered. Training halted.")
            return model

    return model

# Save the trained model as a checkpoint
def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units=None, learning_rate=None, epochs=None):
    """
    Saves a trained model as a checkpoint for later use.
    
    Args:
        model (torch.nn.Module): The trained model to be saved.
        save_dir (str): The directory to save the checkpoint.
        class_to_idx (dict): Mapping of class labels to indices.
        arch (str): Model architecture used (e.g., 'vgg16', 'resnet50').
        hidden_units (int, optional): Number of hidden units in the classifier.
        learning_rate (float, optional): Learning rate used in training.
        epochs (int, optional): Number of training epochs.
    """
    checkpoint = {
        'arch': arch,  
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'classifier': model.classifier
    }

    # Add optional hyperparameters if provided
    if hidden_units is not None:
        checkpoint['hidden_units'] = hidden_units
    if learning_rate is not None:
        checkpoint['learning_rate'] = learning_rate
    if epochs is not None:
        checkpoint['epochs'] = epochs

    # Save the checkpoint
    checkpoint_path = f"{save_dir}/checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
