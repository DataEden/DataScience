import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

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
def train_model(dataloaders, arch='vgg16', hidden_units=512, learning_rate=0.001, epochs=10, use_gpu=False):
    # Load a pre-trained model
    model = getattr(models, arch)(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False

    # Define a new classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move to GPU if available
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                val_loss += criterion(log_ps, labels).item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {val_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

    return model

# Save the trained model
def save_checkpoint(model, save_dir, class_to_idx):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'classifier': model.classifier
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print(f"Model saved to {save_dir}/checkpoint.pth")
