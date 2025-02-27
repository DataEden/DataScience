import torch
from torchvision import models
from PIL import Image
import numpy as np

# Load the model from a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))

    # Retrieve the architecture from the checkpoint
    arch = checkpoint.get('arch', 'vgg16')  # Default to 'vgg16' if not found

    # Dynamically load the correct model architecture
    if hasattr(models, arch):
        model = getattr(models, arch)(pretrained=True)
    else:
        raise ValueError(f"Unknown architecture {arch}. please enter a valid torchvision model.")

    # Load classifier and model state
    #model.classifier = checkpoint['classifier']
    
    if hasattr(model, "classifier"):  
        model.classifier = checkpoint['classifier']  # For VGG, DenseNet
    elif hasattr(model, "fc"):  
        model.fc = checkpoint['fc']  # Restore fc for ResNet
    else:
        raise ValueError(f"Unexpected model architecture: {arch}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint.get('class_to_idx', {})

    print(f"Model loaded from checkpoint: {filepath} using architecture: {arch}")
    return model


# Preprocess the input image
def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

# Predict the top K classes
def predict(image_path, model, top_k=5, use_gpu=False):
    model.eval()
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k, dim=1)

    return top_p.cpu().numpy().flatten(), top_class.cpu().numpy().flatten()
