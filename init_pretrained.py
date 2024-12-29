import torch
import os
from modeling import ImageEncoder
from args import parse_arguments

def main():
    args = parse_arguments()
    print("Initializing pre-trained model...")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.save, exist_ok=True)
    
    # Initialize the ImageEncoder with CLIP weights
    model = ImageEncoder(args)
    
    # Save the pre-trained weights
    save_path = f"{args.save}/pretrained.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved pre-trained model to {save_path}")

if __name__ == "__main__":
    main()