import argparse
from predict_utils import load_checkpoint, process_image, predict

# Parse arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    return parser.parse_args()

def main():
    args = get_input_args()
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k, args.gpu)
    print(f"Top {args.top_k} predictions: {probs}, {classes}")

if __name__ == "__main__":
    main()
