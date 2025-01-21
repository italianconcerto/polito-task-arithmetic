import json
import random
import torch
from tqdm import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from utils import DotDict, train_diag_fim_logtr

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


def evaluate_singletask_model(args, datasets, task_vectors, alpha, pretrained_path):
    results = {}
    
    for dataset_name in tqdm(datasets):
        print(f"\nEvaluating singletask {dataset_name} scaled with alpha {alpha}")
        task_vector = task_vectors[dataset_name]
        # pretrained_model = open(pretrained_path, "rb")
        # Maybe should do this?
        #----
        # merged_encoder = pretrained_model
        # for vector in task_vectors.values():
        #     merged_encoder = vector.apply_to(merged_encoder, scaling_coef=alpha)
        #----
        encoder = task_vector.apply_to(pretrained_path, scaling_coef=alpha)
        encoder = encoder.to(args.device)
        # pretrained_model.close()
        encoder.eval()
        
        
        head = get_classification_head(args, f"{dataset_name}Val")
        head = head.to(args.device)
        model = ImageClassifier(encoder, head)
        model = model.to(args.device)
        model.eval()  

        # Evaluate on train set
        train_dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=model.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        train_loader = get_dataloader(train_dataset, is_train=True, args=args)
        train_acc = evaluate(model, train_loader, args)
        
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
        
        fim_logtr = train_diag_fim_logtr(args, model, dataset_name + "Val")
        
        results[dataset_name] = {
            "scaled_train_acc": train_acc,
            "scaled_val_acc": val_acc,
            "scaled_test_acc": test_acc,
            "fim_logtr": fim_logtr
        }
        

    return results
        
        


def evaluate_multitask_model(args, datasets, task_vectors, alpha, pretrained_path):
    results = {}
    
    print(f"\nEvaluating alpha = {alpha:.2f}")
    
    # Combine all task vectors at once using sum()
    combined_vector = sum(task_vectors.values(), start=None)
        
    # Apply the combined vector to get merged encoder
    # pretrained_model = open(pretrained_path, "rb")
    # Maybe should do this?
    #----
    # merged_encoder = pretrained_model
    # for vector in task_vectors.values():
    #     merged_encoder = vector.apply_to(merged_encoder, scaling_coef=alpha)
    #----
    merged_encoder = combined_vector.apply_to(pretrained_path, scaling_coef=alpha)
    merged_encoder = merged_encoder.to(args.device)
    # pretrained_model.close()
    merged_encoder.eval()
    
    # Load single task accuracies for normalization
    with open(f"{args.save}/single_task_results.json", 'r') as f:
        single_task_results = json.load(f)
    
    # Evaluate on each dataset
    test_absolute_accs = []
    train_absolute_accs = []
    val_absolute_accs = []
    val_normalized_accs = []
    train_normalized_accs = []
    test_normalized_accs = []
    fim_logtrs = []
    
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")
        # Create model with the merged encoder and dataset-specific head
        head = get_classification_head(args, f"{dataset_name}Val")
        head = head.to(args.device)
        model = ImageClassifier(merged_encoder, head)
        model = model.to(args.device)
        model.eval()  # Ensure evaluation mode
        
        # Get single-task accuracy for normalization
        single_task_val_acc = single_task_results[dataset_name]["val_acc"]
        single_task_train_acc = single_task_results[dataset_name]["train_acc"]
        single_task_test_acc = single_task_results[dataset_name]["test_acc"]
        
        # Evaluate on validation set
        train_dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=model.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        train_loader = get_dataloader(train_dataset, is_train=True, args=args)
        train_acc = evaluate(model, train_loader, args, desc=f"Training {dataset_name}")
        train_absolute_accs.append(train_acc)

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
        val_absolute_accs.append(val_acc)
        
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
        test_absolute_accs.append(test_acc)
        
        train_normalized_accs.append(train_acc / single_task_train_acc)
        val_normalized_accs.append(val_acc / single_task_val_acc)
        test_normalized_accs.append(test_acc / single_task_test_acc)
        
        fim_logtr = train_diag_fim_logtr(args, model, dataset_name+"Val")
        fim_logtrs.append(fim_logtr)
        
        results[dataset_name]['absolute'] = {
            "train": train_acc,
            "val": val_acc,
            "test": test_acc,
            "fim_logtr": fim_logtr
        }
        
        results[dataset_name]['normalized'] = {
            "train": train_acc / single_task_train_acc,
            "val": val_acc / single_task_val_acc,
            "test": test_acc / single_task_test_acc,
        }
        # Print current results
        print(f"{dataset_name} Results:")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        print(f"  Normalized Acc: {(test_acc/single_task_val_acc):.4f}")
    
    
    results['average']['normalized'] = {
        "val": val_normalized_accs,
        "train": train_normalized_accs,
        "test": test_normalized_accs,
    }
    results["average"]["absolute"] = {
        "test": test_absolute_accs,
        "train": train_absolute_accs,
        "val": val_absolute_accs,
        "fim_logtr": fim_logtrs,
    }
    
    # Calculate averages
    avg_absolute_acc = sum(test_absolute_accs) / len(test_absolute_accs)
    avg_val_normalized_acc = sum(val_normalized_accs) / len(val_normalized_accs)
    avg_train_normalized_acc = sum(train_normalized_accs) / len(train_normalized_accs)
    avg_test_normalized_acc = sum(test_normalized_accs) / len(test_normalized_accs)
    avg_fim_logtr = sum(fim_logtrs) / len(fim_logtrs)
    
    
    # results["average"] = {
    #     "avg_absolute_acc": avg_absolute_acc,
    #     "avg_val_normalized_acc": avg_val_normalized_acc,
    #     "avg_train_normalized_acc": avg_train_normalized_acc,
    #     "avg_test_normalized_acc": avg_test_normalized_acc,
    #     "normalized_acc": avg_val_normalized_acc,
    #     "avg_fim_logtr": avg_fim_logtr
    # }
    
    print(f"\nAverage Results for alpha = {alpha:.2f}:")
    print(f"  Absolute Acc: {avg_absolute_acc:.2f}%")
    print(f"  Normalized Acc (from Validation): {avg_val_normalized_acc:.4f}")
    print(f"  Average FIM Log-Trace: {avg_fim_logtr:.4f}")
    print(f"  Average Train Acc: {avg_train_normalized_acc:.4f}")
    print(f"  Average Val Acc: {avg_val_normalized_acc:.4f}")
    print(f"  Average Test Acc: {avg_test_normalized_acc:.4f}")
    
    return results

def main():
    args = parse_arguments()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
    
    # Set device to CUDA if available, CPU as fallback
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print("Using CUDA device")
    else:
        args.device = torch.device('cpu')
        print("Using CPU device")
    
    datasets = args.eval_datasets
    
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
    all_results = []
    
    for alpha in tqdm(alphas, desc="Testing alphas"):
        results = evaluate_multitask_model(
            args=args,
            datasets=datasets,
            task_vectors=task_vectors,
            alpha=alpha,
            pretrained_path=pretrained_path
        )
        all_results.append(results)
        avg_normalized_acc = results["average"]["normalized_acc"]
        
        assert avg_normalized_acc == results["average"]["avg_val_normalized_acc"]
        
        if avg_normalized_acc > best_avg_normalized_acc:
            best_avg_normalized_acc = avg_normalized_acc
            best_alpha = alpha
            best_results = results
            print(f"\nNew best alpha: {best_alpha:.2f} with normalized acc: {best_avg_normalized_acc:.4f}")
    
    # Save results
    final_results = {
        "best_alpha": best_alpha,
        "best_results": best_results,
        "all_results": all_results
    }
    
    with open(f"{args.save}/task_addition_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
        
    singletask_scaled_results = evaluate_singletask_model(args, datasets, task_vectors, best_alpha, pretrained_path)

    with open(f"{args.save}/single_task_scaled_results.json", "w") as f:
        json.dump(singletask_scaled_results, f, indent=4)
    
    print(f"Saved results to {args.save}/single_task_scaled_results.json")
    print("\nFinal Results:")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Best average normalized accuracy: {best_avg_normalized_acc:.4f}")
    print(f"Best average absolute accuracy: {best_results['average']['absolute_acc']:.4f}")

if __name__ == "__main__":
    main()
