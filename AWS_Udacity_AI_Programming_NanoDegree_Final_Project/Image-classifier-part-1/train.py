import argparse
from train_utils import load_data, train_model, save_checkpoint

# List of available models
AVAILABLE_MODELS = ["vgg16", "resnet50", "densenet121", "mobilenet_v3_large"]

# Parse command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset and save the model.")

    # Required positional argument
    parser.add_argument("data_dir", type=str, help="Directory of the dataset")

    # Optional arguments
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the checkpoint")
    parser.add_argument("--arch", type=str, choices=AVAILABLE_MODELS, help="Model architecture to use")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    return parser.parse_args()

# Main function
def main():
    args = get_input_args()

    # If the user didn't specify an architecture, display available models
    if args.arch is None:
        print("\n Available Architectures for Training ")
        for model in AVAILABLE_MODELS:
            print(f" {model}")
        print("\n Run the command again with `--arch <model>` to train a model. Example:")
        print("   python train.py flowers --arch resnet50 --epochs 3 --gpu\n")
        return

    try:
        print("\n Loading data...")
        dataloaders, class_to_idx = load_data(args.data_dir)

        print(f"\n Training model: {args.arch} with {args.hidden_units} hidden units...")
        model = train_model(dataloaders, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)

        print(f"\n Saving checkpoint to {args.save_dir}/checkpoint.pth...")
        save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.hidden_units, args.learning_rate, args.epochs)

        print("\n Training complete! Model checkpoint saved.\n")

    except Exception as e:
        print(f"\n Error during training: {e}")

# Run script
if __name__ == "__main__":
    main()
