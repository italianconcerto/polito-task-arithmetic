from typing import Dict, Optional, Union, List
import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from argparse import Namespace
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import train_diag_fim_logtr
import os

def evaluate(
    model: ImageClassifier,
    data_loader: torch.utils.data.DataLoader,
    args: Namespace,
    split: str
) -> Dict[str, float]:
    model.eval()
    correct: int = 0
    total: int = 0
    total_loss: float = 0
    criterion = nn.CrossEntropyLoss()
    
    all_predictions: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f'Evaluating {split}')
        for batch in progress_bar:
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
            
            progress_bar.set_postfix({
                'acc': 100. * correct / total,
                'loss': total_loss / (progress_bar.n + 1)
            })
    
    accuracy: float = 100. * correct / total
    avg_loss: float = total_loss / len(data_loader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'num_samples': total
    }

def evaluate_single_task(
    args: Namespace,
    dataset_name: str,
    model_path: Optional[str],
    save: bool = False
) -> Dict[str, Dict]:
    # Initialize model
    encoder: ImageEncoder = ImageEncoder(args)
    head = get_classification_head(args, f"{dataset_name}Val")
    model: ImageClassifier = ImageClassifier(encoder, head)
    
    # Load model weights if provided
    print(f"Loading model from {model_path}")
    loaded_encoder = torch.load(model_path, map_location=args.device)
    if isinstance(loaded_encoder, dict):
        model.image_encoder.load_state_dict(loaded_encoder)
    else:
        model.image_encoder = loaded_encoder
    
    model = model.to(args.device)
    model.eval()
    
    results: Dict[str, Dict] = {}
    
    # Evaluate on validation set
    val_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )
    val_loader = get_dataloader(val_dataset, is_train=False, args=args)
    val_results = evaluate(model, val_loader, args, "validation")
    
    # Evaluate on test set
    test_dataset = get_dataset(
        dataset_name,
        preprocess=model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)
    test_results = evaluate(model, test_loader, args, "test")
    
    # Compute Fisher Information Matrix log-trace
    fim_logtr: float = train_diag_fim_logtr(args, model, dataset_name)
    
    # Store all results
    results = {
        'dataset_info': {
            'name': dataset_name,
            'model_path': model_path
        },
        'validation': {
            'accuracy': val_results['accuracy'],
            'loss': val_results['loss'],
            'num_samples': val_results['num_samples'],
            'predictions': val_results['predictions'],
            'labels': val_results['labels']
        },
        'test': {
            'accuracy': test_results['accuracy'],
            'loss': test_results['loss'],
            'num_samples': test_results['num_samples'],
            'predictions': test_results['predictions'],
            'labels': test_results['labels']
        },
        'fim_logtr': fim_logtr
    }
    
    # Print summary
    print(f"\nResults for {dataset_name}:")
    print(f"Validation Accuracy: {val_results['accuracy']:.2f}%")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"FIM Log-Trace: {fim_logtr:.4f}")
    
    # Save results if requested
    if save and args.save:
        # Create results directory if it doesn't exist
        os.makedirs(args.save, exist_ok=True)
        
        # Save detailed results
        results_filename = f"{dataset_name}_eval_results.json"
        if model_path:
            # Extract model identifier from path (e.g., best_accuracy_epoch45)
            model_id = os.path.splitext(os.path.basename(model_path))[0].split('_', 1)[1]
            results_filename = f"{dataset_name}_{model_id}_eval_results.json"
        
        save_path = os.path.join(args.save, results_filename)
        
        # Convert all numeric types to native Python types for JSON serialization
        json_results = {
            'dataset_info': results['dataset_info'],
            'validation': {
                'accuracy': float(results['validation']['accuracy']),
                'loss': float(results['validation']['loss']),
                'num_samples': results['validation']['num_samples'],
                'predictions': results['validation']['predictions'],
                'labels': results['validation']['labels']
            },
            'test': {
                'accuracy': float(results['test']['accuracy']),
                'loss': float(results['test']['loss']),
                'num_samples': results['test']['num_samples'],
                'predictions': results['test']['predictions'],
                'labels': results['test']['labels']
            },
            'fim_logtr': float(results['fim_logtr'])
        }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"\nSaved detailed results to {save_path}")
    
    return results

def main() -> None:
    from args import parse_arguments
    args: Namespace = parse_arguments()
    
    # Example usage for a single dataset
    dataset_name = "DTD"
    model_path = os.path.join(args.save, f"{dataset_name}_best_accuracy_epoch45.pt")  # Example path
    
    results = evaluate_single_task(
        args=args,
        dataset_name=dataset_name,
        model_path=model_path,
        save=True
    )

if __name__ == "__main__":
    main()