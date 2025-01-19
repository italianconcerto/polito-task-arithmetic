from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import json
import torch
from tqdm import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from task_vectors import NonLinearTaskVector
from utils import DotDict, train_diag_fim_logtr
import os

def evaluate(
    model: ImageClassifier,
    loader: torch.utils.data.DataLoader,
    args: Namespace,
    desc: Optional[str] = None
) -> Dict[str, float]:
    model.eval()
    correct: int = 0
    total: int = 0
    total_loss: float = 0
    criterion = nn.CrossEntropyLoss()
    
    all_predictions: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = maybe_dictionarize(batch)
            x: torch.Tensor = batch['images'].to(args.device)
            y: torch.Tensor = batch['labels'].to(args.device)
            
            outputs: torch.Tensor = model(x)
            loss: torch.Tensor = criterion(outputs, y)
            
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            total_loss += loss.item()
            
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    
    accuracy: float = 100. * correct / total
    avg_loss: float = total_loss / len(loader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'num_samples': total
    }

def evaluate_multitask_model(
    args: Namespace,
    task_vectors: Dict[str, NonLinearTaskVector],
    alpha: float,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    datasets = args.eval_dataset
    print(f"\nEvaluating alpha = {alpha:.2f}")
    
    # Combine all task vectors at once using sum()
    combined_vector = sum(task_vectors.values(), start=None)
    
    pretrained_path: str = f"{args.save}/pretrained.pt"
    # Apply the combined vector to get merged encoder
    pretrained_model = torch.load(pretrained_path, map_location=args.device)
    merged_encoder = combined_vector.apply_to(pretrained_model, scaling_coef=alpha)
    merged_encoder = merged_encoder.to(args.device)
    merged_encoder.eval()
    
    # Load single task accuracies for normalization
    with open(f"{args.save}/single_task_results.json", 'r') as f:
        single_task_results = json.load(f)
    
    # Store results for each dataset
    dataset_results: Dict[str, Dict] = {}
    absolute_accs: List[float] = []
    normalized_accs: List[float] = []
    
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}...")
        
        # Create model with the merged encoder and dataset-specific head
        head = get_classification_head(args, f"{dataset_name}Val")
        head = head.to(args.device)
        model = ImageClassifier(merged_encoder, head)
        model = model.to(args.device)
        model.eval()
        
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
        val_results = evaluate(model, val_loader, args, desc=f"Validating {dataset_name}")
        
        # Calculate FIM logtr for validation dataset
        fim_logtr: float = train_diag_fim_logtr(args, model, f"{dataset_name}Val")
        
        # Evaluate on test set
        test_dataset = get_dataset(
            dataset_name,
            preprocess=model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        test_loader = get_dataloader(test_dataset, is_train=False, args=args)
        test_results = evaluate(model, test_loader, args, desc=f"Testing {dataset_name}")
        
        # Store accuracies
        absolute_accs.append(test_results['accuracy'])
        normalized_accs.append(test_results['accuracy'] / single_task_acc)
        
        dataset_results[dataset_name] = {
            'validation': val_results,
            'test': test_results,
            'single_task_acc': single_task_acc,
            'normalized_acc': test_results['accuracy'] / single_task_acc,
            'fim_logtr': fim_logtr
        }
        
        # Print current results
        print(f"{dataset_name} Results:")
        print(f"  Val Acc: {val_results['accuracy']:.2f}%")
        print(f"  Test Acc: {test_results['accuracy']:.2f}%")
        print(f"  Normalized Acc: {(test_results['accuracy']/single_task_acc):.4f}")
        print(f"  FIM logtr: {fim_logtr:.4f}")
    
    # Calculate averages
    avg_absolute_acc = sum(absolute_accs) / len(absolute_accs)
    avg_normalized_acc = sum(normalized_accs) / len(normalized_accs)
    
    results = {
        'alpha': alpha,
        'dataset_results': dataset_results,
        'average_metrics': {
            'absolute_acc': avg_absolute_acc,
            'normalized_acc': avg_normalized_acc
        }
    }
    
    # Save results if requested
    if args.save:
        save_path = os.path.join(args.save, f"multitask_alpha{alpha:.2f}_results.json")
        
        # Convert to JSON-serializable format
        json_results = {
            'alpha': float(alpha),
            'dataset_results': {
                name: {
                    'validation': {
                        'accuracy': float(data['validation']['accuracy']),
                        'loss': float(data['validation']['loss']),
                        'num_samples': data['validation']['num_samples'],
                    },
                    'test': {
                        'accuracy': float(data['test']['accuracy']),
                        'loss': float(data['test']['loss']),
                        'num_samples': data['test']['num_samples'],
                    },
                    'single_task_acc': float(data['single_task_acc']),
                    'normalized_acc': float(data['normalized_acc']),
                    'fim_logtr': float(data['fim_logtr'])
                }
                for name, data in dataset_results.items()
            },
            'average_metrics': {
                'absolute_acc': float(avg_absolute_acc),
                'normalized_acc': float(avg_normalized_acc)
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"\nSaved detailed results to {save_path}")
    
    return results

def main() -> None:
    args = parse_arguments()
    
    # Set device
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
    
    alpha_results = {}
    for alpha in tqdm(alphas, desc="Testing alphas"):
        results = evaluate_multitask_model(
            args=args,
            datasets=datasets,
            task_vectors=task_vectors,
            alpha=alpha,
            pretrained_path=pretrained_path,
            save=True
        )
        
        alpha_results[alpha] = results
        avg_normalized_acc = results['average_metrics']['normalized_acc']
        
        if avg_normalized_acc > best_avg_normalized_acc:
            best_avg_normalized_acc = avg_normalized_acc
            best_alpha = alpha
            best_results = results
            print(f"\nNew best alpha: {best_alpha:.2f} with normalized acc: {best_avg_normalized_acc:.4f}")
    
    # Save final results
    final_results = {
        'best_alpha': best_alpha,
        'alpha_results': alpha_results,
        'best_results': best_results
    }
    
    save_path = os.path.join(args.save, "task_addition_final_results.json")
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print("\nFinal Results:")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Best average normalized accuracy: {best_avg_normalized_acc:.4f}")
    print(f"Best average absolute accuracy: {best_results['average_metrics']['absolute_acc']:.2f}%")
    print(f"\nSaved all results to {save_path}")

if __name__ == "__main__":
    main()
