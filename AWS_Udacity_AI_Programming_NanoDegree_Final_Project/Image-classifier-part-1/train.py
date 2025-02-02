import argparse
from train_utils import load_data, train_model, save_checkpoint

# Parse arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset and save the model.")
    parser.add_argument("data_dir", type=str, help="Directory of the dataset")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (vgg16, vgg13, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    return parser.parse_args()

def main():
    args = get_input_args()
    dataloaders, class_to_idx = load_data(args.data_dir)
    model = train_model(dataloaders, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)
    save_checkpoint(model, args.save_dir, class_to_idx)

if __name__ == "__main__":
    main()
