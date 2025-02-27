import torch

def validate_model(model, dataloaders, criterion, device):
    """
    Validate the trained model using the validation dataset.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataloaders (dict): Dictionary containing the validation DataLoader.
        criterion (torch.nn.Module): The loss function.
        device (str): 'cuda' or 'cpu' for model computation.

    Returns:
        tuple: (validation_loss, validation_accuracy)
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    accuracy = 0

    with torch.no_grad():
        for val_inputs, val_labels in dataloaders['valid']:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            logps = model(val_inputs)
            batch_loss = criterion(logps, val_labels)
            val_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == val_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    val_loss /= len(dataloaders['valid'])
    accuracy /= len(dataloaders['valid'])

    print(f"\nValidation Loss: {val_loss:.3f}  |  Validation Accuracy: {accuracy:.3f}")

    return val_loss, accuracy
