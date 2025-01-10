import json
import torch
from tqdm import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from utils import DotDict

def evaluate(model, loader, args, desc=None):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = maybe_dictionarize(batch)
            x, y = batch['images'].to(args.device), batch['labels'].to(args.device)
            
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    acc = 100. * correct / total
    if desc:
        print(f"{desc}: {acc:.2f}%")
    return acc

def evaluate_multitask_model(args, datasets, task_vectors, alpha, pretrained_path):
    results = {}
    
    print(f"\nEvaluating alpha = {alpha:.2f}")
    
    # Combine all task vectors at once using sum()
    combined_vector = sum(task_vectors.values(), start=None)
        
    # Apply the combined vector to get merged encoder
    pretrained_model = open(pretrained_path, "rb")
    merged_encoder = combined_vector.apply_to(pretrained_model, scaling_coef=alpha)
    merged_encoder = merged_encoder.to(args.device)
    pretrained_model.close()
    merged_encoder.eval()
    
    # Load single task accuracies for normalization
    with open(f"{args.save}/single_task_results.json", 'r') as f:
        single_task_results = json.load(f)
    
    # Evaluate on each dataset
    absolute_accs = []
    normalized_accs = []
    
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")
        # Create model with the merged encoder and dataset-specific head
        head = get_classification_head(args, f"{dataset_name}Val")
        head = head.to(args.device)
        model = ImageClassifier(merged_encoder, head)
        model = model.to(args.device)
        model.eval()  # Ensure evaluation mode
        
        # Get single-task accuracy for normalization
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
        val_acc = evaluate(model, val_loader, args, desc=f"Validating {dataset_name}")
        
        # Evaluate on test set
        test_dataset = get_dataset(
            dataset_name,
            preprocess=model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        test_loader = get_dataloader(test_dataset, is_train=False, args=args)
        test_acc = evaluate(model, test_loader, args, desc=f"Testing {dataset_name}")
        
        # Store accuracies
        absolute_accs.append(test_acc)
        normalized_accs.append(test_acc / single_task_acc)
        
        results[dataset_name] = {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "normalized_acc": test_acc / single_task_acc
        }
        
        # Print current results
        print(f"{dataset_name} Results:")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Normalized Acc: {(test_acc/single_task_acc):.4f}")
    
    # Calculate averages
    avg_absolute_acc = sum(absolute_accs) / len(absolute_accs)
    avg_normalized_acc = sum(normalized_accs) / len(normalized_accs)
    
    results["average"] = {
        "absolute_acc": avg_absolute_acc,
        "normalized_acc": avg_normalized_acc
    }
    
    print(f"\nAverage Results for alpha = {alpha:.2f}:")
    print(f"  Absolute Acc: {avg_absolute_acc:.2f}%")
    print(f"  Normalized Acc: {avg_normalized_acc:.4f}")
    
    return results

def main():
    args = parse_arguments()
    
    # Set device to CUDA if available, CPU as fallback
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print("Using CUDA device")
    else:
        args.device = torch.device('cpu')
        print("Using CPU device")
    
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    pretrained_path = f"{args.save}/pretrained.pt"
    
    print("Building task vectors...")
    task_vectors = {}
    for dataset_name in tqdm(datasets):
        finetuned_path = f"{args.save}/{dataset_name}_finetuned.pt"
        task_vectors[dataset_name] = NonLinearTaskVector(
            pretrained_checkpoint=pretrained_path,
            finetuned_checkpoint=finetuned_path
        )
    
    # Find optimal alpha
    print("\nFinding optimal alpha...")
    alphas = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 in 0.05 steps
    best_alpha = 0
    best_avg_normalized_acc = 0
    best_results = None
    
    for alpha in tqdm(alphas, desc="Testing alphas"):
        results = evaluate_multitask_model(
            args=args,
            datasets=datasets,
            task_vectors=task_vectors,
            alpha=alpha,
            pretrained_path=pretrained_path
        )
        
        avg_normalized_acc = results["average"]["normalized_acc"]
        
        if avg_normalized_acc > best_avg_normalized_acc:
            best_avg_normalized_acc = avg_normalized_acc
            best_alpha = alpha
            best_results = results
            print(f"\nNew best alpha: {best_alpha:.2f} with normalized acc: {best_avg_normalized_acc:.4f}")
    
    # Save results
    final_results = {
        "best_alpha": best_alpha,
        "results": best_results
    }
    
    with open(f"{args.save}/task_addition_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\nFinal Results:")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Best average normalized accuracy: {best_avg_normalized_acc:.4f}")
    print(f"Best average absolute accuracy: {best_results['average']['absolute_acc']:.4f}")

if __name__ == "__main__":
    main()
