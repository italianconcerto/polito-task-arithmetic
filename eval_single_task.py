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

def evaluate_models(args: Namespace, finetuning_results: Dict[str, Dict]) -> Dict[str, Dict]:
    evaluation_results: Dict[str, Dict] = {}
    
    for dataset_name, dataset_results in finetuning_results.items():
        print(f"\nEvaluating models for {dataset_name}")
        dataset_eval_results = {}
        
        # Get paths for all model versions
        model_paths = {
            'best_accuracy': dataset_results['best_metrics']['accuracy']['save_path'],
            'best_fim': dataset_results['best_metrics']['fim_logtr']['save_path'],
            'final': dataset_results['final_model']['save_path']
        }
        
        # Evaluate each model version
        for model_type, model_path in model_paths.items():
            if model_path and os.path.exists(model_path):
                print(f"Evaluating {model_type} model: {model_path}")
                
                # Initialize model
                encoder: ImageEncoder = ImageEncoder(args)
                head = get_classification_head(args, f"{dataset_name}Val")
                model: ImageClassifier = ImageClassifier(encoder, head)
                
                # Load model weights
                loaded_encoder = torch.load(model_path, map_location=args.device)
                if isinstance(loaded_encoder, dict):
                    model.image_encoder.load_state_dict(loaded_encoder)
                else:
                    model.image_encoder = loaded_encoder
                
                model = model.to(args.device)
                model.eval()
                
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
                
                # Store results for this model version
                dataset_eval_results[model_type] = {
                    'model_info': {
                        'type': model_type,
                        'path': model_path
                    },
                    'validation': val_results,
                    'test': test_results,
                    'fim_logtr': fim_logtr
                }
                
                # Print summary
                print(f"\nResults for {model_type} model:")
                print(f"Validation Accuracy: {val_results['accuracy']:.2f}%")
                print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
                print(f"FIM Log-Trace: {fim_logtr:.4f}")
        
        evaluation_results[dataset_name] = dataset_eval_results
    
    # Save results if requested
    if args.save:
        save_path = os.path.join(args.save, "evaluation_results.json")
        
        # Convert numeric types to native Python types for JSON serialization
        json_results = {}
        for dataset_name, dataset_results in evaluation_results.items():
            json_results[dataset_name] = {}
            for model_type, model_results in dataset_results.items():
                json_results[dataset_name][model_type] = {
                    'model_info': model_results['model_info'],
                    'validation': {
                        'accuracy': float(model_results['validation']['accuracy']),
                        'loss': float(model_results['validation']['loss']),
                        'num_samples': model_results['validation']['num_samples'],
                        'predictions': model_results['validation']['predictions'],
                        'labels': model_results['validation']['labels']
                    },
                    'test': {
                        'accuracy': float(model_results['test']['accuracy']),
                        'loss': float(model_results['test']['loss']),
                        'num_samples': model_results['test']['num_samples'],
                        'predictions': model_results['test']['predictions'],
                        'labels': model_results['test']['labels']
                    },
                    'fim_logtr': float(model_results['fim_logtr'])
                }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"\nSaved evaluation results to {save_path}")
    
    return evaluation_results

def main() -> None:
    from args import parse_arguments
    args: Namespace = parse_arguments()
    
    # Load finetuning results if they exist
    finetuning_results_path = os.path.join(args.save, "finetuning_results.json")
    if os.path.exists(finetuning_results_path):
        with open(finetuning_results_path, 'r') as f:
            finetuning_results = json.load(f)
    else:
        raise FileNotFoundError(f"Finetuning results not found at {finetuning_results_path}")
    
    evaluation_results = evaluate_models(args, finetuning_results)

if __name__ == "__main__":
    main()