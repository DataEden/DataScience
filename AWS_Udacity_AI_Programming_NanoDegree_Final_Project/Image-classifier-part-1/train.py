import argparse
from train_utils import load_data, train_model, save_checkpoint

# Parse command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset and save the model.")
    
    # Required positional argument
    parser.add_argument("data_dir", type=str, help="Directory of the dataset")
    
    # Optional arguments
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (vgg16, vgg13, resnet50, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    return parser.parse_args()

# Main function
def main():
    args = get_input_args()

    try:
        print("\n Loading data...")
        dataloaders, class_to_idx = load_data(args.data_dir)
        
        print(f" Training model: {args.arch} with {args.hidden_units} hidden units...")
        model = train_model(dataloaders, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)
        
        print(f"\n Saving checkpoint to {args.save_dir}/checkpoint.pth...")
        save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.hidden_units, args.learning_rate, args.epochs)
        
        print("\n Training complete! Model checkpoint saved.\n")

    except Exception as e:
        print(f"\n Error during training: {e}")

# Run script
if __name__ == "__main__":
    main()
