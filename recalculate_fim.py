import os
import json
import torch
from tqdm.auto import tqdm
from args import parse_arguments
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import train_diag_fim_logtr
from datasets.registry import get_dataset

def recalculate_fim_single_task(args, dataset_name):
    """Recalculate FIM log-trace for a single task model."""
    print(f"\nRecalculating FIM for {dataset_name}")
    
    # Initialize model
    encoder = ImageEncoder(args)
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    
    # Load fine-tuned weights
    checkpoint_path = f"{args.save}/{dataset_name}_finetuned.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Could not find checkpoint at {checkpoint_path}")
        return None
        
    loaded_encoder = torch.load(checkpoint_path, map_location=args.device)
    model.image_encoder = loaded_encoder
    model = model.to(args.device)
    
    # Calculate FIM log-trace
    fim_logtr = train_diag_fim_logtr(args, model, dataset_name)
    print(f"FIM Log-Trace: {fim_logtr:.4f}")
    
    return fim_logtr

def recalculate_fim_multitask(args, datasets, task_vectors, alpha, pretrained_path):
    """Recalculate FIM log-trace for multitask model at given alpha."""
    print(f"\nRecalculating FIM for multitask model with alpha={alpha}")
    
    # Combine task vectors
    combined_vector = sum(task_vectors.values(), start=None)
    
    # Apply the combined vector to get merged encoder
    with open(pretrained_path, "rb") as f:
        pretrained_model = torch.load(f, map_location=args.device)
    merged_encoder = combined_vector.apply_to(pretrained_model, scaling_coef=alpha)
    merged_encoder = merged_encoder.to(args.device)
    merged_encoder.eval()
    
    fim_results = {}
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        head = get_classification_head(args, f"{dataset_name}Val")
        head = head.to(args.device)
        model = ImageClassifier(merged_encoder, head)
        model = model.to(args.device)
        
        fim_logtr = train_diag_fim_logtr(args, model, dataset_name)
        fim_results[dataset_name] = fim_logtr
        print(f"FIM Log-Trace: {fim_logtr:.4f}")
    
    return fim_results

def main():
    args = parse_arguments()
    
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Using CUDA device")
    else:
        args.device = torch.device('cpu')
        print("Using CPU device")
    
    # Get list of datasets
    datasets = args.eval_datasets
    pretrained_path = f"{args.save}/pretrained.pt"
    
    # Recalculate single task FIM values
    print("Processing single task models...")
    single_task_results = {}
    
    for dataset_name in datasets:
        fim_logtr = recalculate_fim_single_task(args, dataset_name)
        if fim_logtr is not None:
            single_task_results[dataset_name] = {'fim_logtr': fim_logtr}
    
    # Save single task results
    with open(f"{args.save}/single_task_fim_results.json", 'w') as f:
        json.dump(single_task_results, f, indent=4)
    
    # Get best alpha from previous results
    try:
        with open(f"{args.save}/task_addition_results.json", 'r') as f:
            previous_results = json.load(f)
            best_alpha = previous_results.get('best_alpha')
    except FileNotFoundError:
        print("Warning: Could not find previous task addition results")
        best_alpha = None
    
    if best_alpha is not None:
        print(f"\nProcessing multitask model at best alpha ({best_alpha})...")
        
        # Build task vectors
        from task_vectors import NonLinearTaskVector
        task_vectors = {}
        for dataset_name in datasets:
            finetuned_path = f"{args.save}/{dataset_name}_finetuned.pt"
            if os.path.exists(finetuned_path):
                task_vectors[dataset_name] = NonLinearTaskVector(
                    pretrained_checkpoint=pretrained_path,
                    finetuned_checkpoint=finetuned_path
                )
        
        # Recalculate multitask FIM values
        multitask_fim_results = recalculate_fim_multitask(
            args,
            datasets,
            task_vectors,
            best_alpha,
            pretrained_path
        )
        
        # Save multitask results
        with open(f"{args.save}/multitask_fim_results.json", 'w') as f:
            json.dump({
                'alpha': best_alpha,
                'results': multitask_fim_results
            }, f, indent=4)
    
    print("\nFIM recalculation complete!")

if __name__ == "__main__":
    main()