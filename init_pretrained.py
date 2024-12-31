import torch
import os
from modeling import ImageEncoder
from args import parse_arguments

def main():
    args = parse_arguments()
    print("Initializing pre-trained model...")
    
    os.makedirs(args.save, exist_ok=True)
    
    model = ImageEncoder(args)
    save_path = f"{args.save}/pretrained.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved pre-trained model state dict to {save_path}")

if __name__ == "__main__":
    main()