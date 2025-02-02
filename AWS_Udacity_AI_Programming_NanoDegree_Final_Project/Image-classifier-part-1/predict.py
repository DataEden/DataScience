import argparse
import torch
import json
from predict_utils import load_checkpoint, process_image, predict

# Parse command-line arguments**
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
    
    # Positional arguments (required)
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")

    # Optional arguments
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    return parser.parse_args()

# Main prediction function**
def main():
    args = get_input_args()

    # Load the trained model from checkpoint**
    model = load_checkpoint(args.checkpoint)

    # Set device (GPU if available and selected)**
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process input image**
    image_tensor = process_image(args.input)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Make predictions**
    probs, classes = predict(image_tensor, model, args.top_k)

    # Map class indices to actual category names if provided**
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]
    else:
        class_names = classes  # Default to numerical class labels

    # Print predictions
    print("\n Prediction Results")
    for i in range(args.top_k):
        print(f"{i+1}. {class_names[i]} ({probs[i]*100:.2f}%)")

if __name__ == "__main__":
    main()
