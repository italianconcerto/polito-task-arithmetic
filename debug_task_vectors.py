import torch
from args import parse_arguments
from task_vectors import NonLinearTaskVector
from modeling import ImageEncoder
from utils import DotDict

def main():
    args = parse_arguments()
    datasets = ["MNIST", "SVHN"]  # Test with just 2 datasets
    pretrained_path = f"{args.save}/pretrained.pt"
    
    # Load task vectors
    task_vectors = {}
    for dataset_name in datasets:
        finetuned_path = f"{args.save}/{dataset_name}_finetuned.pt"
        task_vector = NonLinearTaskVector(
            pretrained_checkpoint=pretrained_path,
            finetuned_checkpoint=finetuned_path
        )
        task_vectors[dataset_name] = task_vector
        
        # Print vector stats
        print(f"\n{dataset_name} task vector stats:")
        total_params = 0
        total_magnitude = 0
        for k, v in task_vector.vector.items():
            params = v.numel()
            magnitude = torch.norm(v).item()
            total_params += params
            total_magnitude += magnitude
            print(f"{k}: {params} params, magnitude: {magnitude:.4f}")
        print(f"Total params: {total_params}")
        print(f"Total magnitude: {total_magnitude:.4f}")
    
    # Combine vectors
    combined = None
    for dataset_name in datasets:
        if combined is None:
            combined = task_vectors[dataset_name]
        else:
            combined = combined + task_vectors[dataset_name]
    
    # Print combined stats
    print("\nCombined vector stats:")
    total_params = 0
    total_magnitude = 0
    for k, v in combined.vector.items():
        params = v.numel()
        magnitude = torch.norm(v).item()
        total_params += params
        total_magnitude += magnitude
        print(f"{k}: {params} params, magnitude: {magnitude:.4f}")
    print(f"Total params: {total_params}")
    print(f"Total magnitude: {total_magnitude:.4f}")
    
    # Test applying with different alphas
    print("\nTesting different alphas...")
    for alpha in [0.3, 0.5, 0.7]:
        print(f"\nAlpha {alpha}:")
        
        # Create fresh encoder
        # Set device
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS device")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Using CUDA device")
        else:
            device = 'cpu'
            print("Using CPU device")
            
        encoder_args = DotDict({
            'model': 'ViT-B-32',
            'device': device,
            'openclip_cachedir': getattr(args, 'openclip_cachedir', None),
            'cache_dir': getattr(args, 'cache_dir', None)
        })
        merged_encoder = ImageEncoder(encoder_args)
        
        # Load pretrained state
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        if not isinstance(pretrained_state, dict):
            pretrained_state = pretrained_state.state_dict()
        
        # Apply task vector
        new_state = {}
        for key in pretrained_state:
            if key in combined.vector:
                new_state[key] = pretrained_state[key] + alpha * combined.vector[key]
            else:
                new_state[key] = pretrained_state[key]
        
        # Load modified state
        merged_encoder.load_state_dict(new_state)
        merged_dict = merged_encoder.state_dict()
        
        total_diff = 0
        for k in pretrained_state:
            if k in merged_dict:
                diff = torch.norm(merged_dict[k] - pretrained_state[k]).item()
                total_diff += diff
                if diff > 0:
                    print(f"{k}: diff magnitude = {diff:.4f}")
        print(f"Total weight difference: {total_diff:.4f}")

if __name__ == "__main__":
    main()
