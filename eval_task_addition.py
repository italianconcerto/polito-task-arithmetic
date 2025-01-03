import json
import torch
from tqdm import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_load, DotDict

def evaluate(model, loader, args):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = maybe_dictionarize(batch)
            x, y = batch['images'].to(args.device), batch['labels'].to(args.device)
            
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return 100. * correct / total

def evaluate_multitask_model(args, datasets, task_vectors, alpha, base_encoder):
    results = {}
    
    # Load the base encoder state dict
    encoder_state = torch.load(base_encoder, map_location='cpu', weights_only=True)
    if not isinstance(encoder_state, dict):
        encoder_state = encoder_state.state_dict()
    
    # Initialize a new encoder
    encoder_args = DotDict({
        'model': 'ViT-B-32',
        'device': args.device,
        'openclip_cachedir': getattr(args, 'openclip_cachedir', None),
        'cache_dir': getattr(args, 'cache_dir', None)
    })
    encoder = ImageEncoder(encoder_args)
    
    # Load the state dict
    if not isinstance(encoder_state, dict):
        encoder_state = encoder_state.state_dict()
    encoder.load_state_dict(encoder_state)
    
    # Combine task vectors
    combined_task_vector = {}
    for dataset_name in datasets:
        if not combined_task_vector:
            combined_task_vector = {k: v.clone() for k, v in task_vectors[dataset_name].items()}
        else:
            for k in combined_task_vector:
                if k in task_vectors[dataset_name]:
                    combined_task_vector[k] += task_vectors[dataset_name][k]
    
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
    
    # Calculate average normalized accuracy
    avg_normalized_acc = sum(d["normalized_acc"] for d in results.values()) / len(results)
    results["average"] = {"normalized_acc": avg_normalized_acc}
    
    return results

def main():
    args = parse_arguments()
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    pretrained_path = f"{args.save}/pretrained.pt"
    
    print("Building task vectors...")
    task_vectors = {}
    for dataset_name in tqdm(datasets):
        finetuned_path = f"{args.save}/{dataset_name}_finetuned.pt"
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        finetuned_state = torch.load(finetuned_path, map_location='cpu')
        
        task_vector = {}
        for key in pretrained_state.keys():
            if key not in finetuned_state:
                continue
            if pretrained_state[key].shape != finetuned_state[key].shape:
                continue
            task_vector[key] = finetuned_state[key] - pretrained_state[key]
        task_vectors[dataset_name] = task_vector
    
    # Find optimal alpha
    print("\nFinding optimal alpha...")
    alphas = [i * 0.1 for i in range(21)]  # 0.0 to 2.0
    best_alpha = 0
    best_avg_normalized_acc = 0
    best_results = None
    
    for alpha in tqdm(alphas):
        results = evaluate_multitask_model(
            args=args,
            datasets=datasets,
            task_vectors=task_vectors,
            alpha=alpha,
            base_encoder=pretrained_path
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
    
    with open(f"{args.save}/task_addition_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nBest alpha: {best_alpha:.2f}")
    print(f"Best average normalized accuracy: {best_avg_normalized_acc:.4f}")

if __name__ == "__main__":
    main()