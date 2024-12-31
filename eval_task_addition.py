import json
import numpy as np
import torch
from tqdm.auto import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from utils import torch_load, DotDict

def evaluate(model, data_loader, args):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = maybe_dictionarize(batch)
            x, y = batch['images'].to(args.device), batch['labels'].to(args.device)
            
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return 100. * correct / total

def evaluate_multitask_model(args, datasets, task_vectors, alpha, base_encoder):
    results = {}
    
    # Load the base encoder
    encoder_args = DotDict({
        'model': 'ViT-B-32',
        'device': args.device,
        'openclip_cachedir': args.openclip_cachedir,
        'cache_dir': args.cache_dir
    })
    encoder = torch_load(base_encoder, device=args.device, encoder_args=encoder_args)
    
    # Combine task vectors
    combined_task_vector = None
    for dataset_name in datasets:
        if combined_task_vector is None:
            combined_task_vector = task_vectors[dataset_name].vector.clone()
        else:
            combined_task_vector += task_vectors[dataset_name].vector
    
    # Apply the combined task vector
    with torch.no_grad():
        for name, param in encoder.named_parameters():
            if name in combined_task_vector:
                param.data += alpha * combined_task_vector[name]
    
    # Evaluate on each dataset
    for dataset_name in datasets:
        # Create model with the modified encoder
        head = get_classification_head(args, f"{dataset_name}Val")
        model = ImageClassifier(encoder, head)
        model = model.to(args.device)
        
        # Get single-task accuracy for normalization
        single_task_results = json.load(open(f"{args.save}/single_task_results.json"))
        single_task_acc = single_task_results[dataset_name]["test_acc"]
        
        # Evaluate on validation set
        val_dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        val_loader = get_dataloader(val_dataset, is_train=False, args=args)
        val_acc = evaluate(model, val_loader, args)
        
        # Evaluate on test set
        test_dataset = get_dataset(
            dataset_name,
            preprocess=model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        test_loader = get_dataloader(test_dataset, is_train=False, args=args)
        test_acc = evaluate(model, test_loader, args)
        
        # Calculate normalized accuracy
        normalized_acc = test_acc / single_task_acc
        
        results[dataset_name] = {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "normalized_acc": normalized_acc
        }
    
    # Calculate average metrics
    avg_test_acc = np.mean([v["test_acc"] for v in results.values()])
    avg_normalized_acc = np.mean([v["normalized_acc"] for v in results.values()])
    
    results["average"] = {
        "test_acc": avg_test_acc,
        "normalized_acc": avg_normalized_acc
    }
    
    return results

def main():
    args = parse_arguments()
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    # Load pretrained model path
    pretrained_path = f"{args.save}/pretrained.pt"
    
    # Build task vectors
    print("Building task vectors...")
    task_vectors = {}
    for dataset_name in tqdm(datasets):
        finetuned_path = f"{args.save}/{dataset_name}_finetuned.pt"
        task_vectors[dataset_name] = NonLinearTaskVector(
            pretrained_path, finetuned_path
        )
    
    # Try different alpha values
    alpha_values = np.arange(0.0, 1.05, 0.05)
    best_results = None
    best_alpha = None
    best_avg_normalized_acc = 0
    
    print("\nFinding optimal alpha...")
    for alpha in tqdm(alpha_values):
        results = evaluate_multitask_model(
            args, datasets, task_vectors, alpha, pretrained_path
        )
        
        avg_normalized_acc = results["average"]["normalized_acc"]
        print(f"\nAlpha: {alpha:.2f}, Avg Normalized Acc: {avg_normalized_acc:.4f}")
        
        if avg_normalized_acc > best_avg_normalized_acc:
            best_avg_normalized_acc = avg_normalized_acc
            best_alpha = alpha
            best_results = results
    
    # Save results
    final_results = {
        "best_alpha": best_alpha,
        "results": best_results
    }
    
    save_path = f"{args.save}/task_addition_results.json"
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nBest alpha: {best_alpha:.2f}")
    print(f"Best average normalized accuracy: {best_avg_normalized_acc:.4f}")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()