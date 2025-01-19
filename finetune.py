from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from argparse import Namespace
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import train_diag_fim_logtr
import copy
import os
import json

def train_one_epoch(
    model: ImageClassifier,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args: Namespace
) -> Tuple[float, float, float, float]:
    # Training phase
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        batch = maybe_dictionarize(batch)
        x: torch.Tensor = batch['images'].to(args.device)
        y: torch.Tensor = batch['labels'].to(args.device)
        
        optimizer.zero_grad()
        outputs: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'train_acc': 100. * correct / total
        })
    
    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Validation phase
    model.eval()
    val_loss: float = 0
    val_correct: int = 0
    val_total: int = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            batch = maybe_dictionarize(batch)
            x: torch.Tensor = batch['images'].to(args.device)
            y: torch.Tensor = batch['labels'].to(args.device)
            
            outputs: torch.Tensor = model(x)
            loss: torch.Tensor = criterion(outputs, y)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += y.size(0)
            val_correct += predicted.eq(y).sum().item()
    
    val_loss = val_loss / len(validation_loader)
    val_acc = 100. * val_correct / val_total
    
    return train_loss, train_acc, val_loss, val_acc

def finetune_model(
    args: Namespace,
    # dataset_name: Optional[str] = None,
    # save: bool = False
) -> Dict[str, Dict]:
    default_epochs_mapping: Dict[str, int] = {
        # "DTD": 76,
        # "EuroSAT": 12,
        # "GTSRB": 11,
        # "MNIST": 5,
        # "RESISC45": 15,
        # "SVHN": 4
        "DTD": 1,
        "EuroSAT": 1,
        "GTSRB": 1,
        "MNIST": 1,
        "RESISC45": 1,
        "SVHN": 1
    }
    dataset_name = args.train_dataset
    datasets_to_process: List[str] = dataset_name if dataset_name else list(default_epochs_mapping.keys())
    
    results: Dict[str, Dict] = {}
    
    for current_dataset in datasets_to_process:
        print(f"\nFine-tuning on {current_dataset}")
        
        encoder: ImageEncoder = ImageEncoder(args)
        head = get_classification_head(args, f"{current_dataset}Val")
        model: ImageClassifier = ImageClassifier(encoder, head)
        model.freeze_head()
        model = model.to(args.device)
        
        optimizer: torch.optim.Optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.wd
        )
        criterion: nn.Module = nn.CrossEntropyLoss()
        
        dataset = get_dataset(
            f"{current_dataset}Val",
            preprocess=model.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        train_loader: torch.utils.data.DataLoader = get_dataloader(dataset, is_train=True, args=args)
        validation_loader: torch.utils.data.DataLoader = get_dataloader(dataset, is_train=False, args=args)
        
        num_epochs: int = args.epochs if args.epochs else default_epochs_mapping[current_dataset]
        
        epoch_results: List[Dict[str, Union[int, float]]] = []
        
        # Track best models and their metrics
        best_val_accuracy: float = float('-inf')
        best_fim_logtr: float = float('-inf')
        best_accuracy_model: Optional[ImageEncoder] = None
        best_fim_model: Optional[ImageEncoder] = None
        best_accuracy_epoch: int = 0
        best_fim_epoch: int = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training and validation step
            train_loss, train_acc, val_loss, val_acc = train_one_epoch(
                model, train_loader, validation_loader, optimizer, criterion, args
            )
            
            # Calculate FIM log trace
            fim_logtr: float = train_diag_fim_logtr(args, model, current_dataset)
            
            # Update best models if needed
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_accuracy_model = copy.deepcopy(model.image_encoder)
                best_accuracy_epoch = epoch + 1
                
            if fim_logtr > best_fim_logtr:
                best_fim_logtr = fim_logtr
                best_fim_model = copy.deepcopy(model.image_encoder)
                best_fim_epoch = epoch + 1
            
            # Record metrics
            epoch_data = {
                'epoch': epoch + 1,
                'loss': train_loss,
                'train_accuracy': train_acc,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc,
                'fim_logtr': fim_logtr,
            }
            epoch_results.append(epoch_data)
            
            print(
                f"Training Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_acc:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_acc:.2f}%, "
                f"FIM Log Trace: {fim_logtr:.4f}"
            )
        
        # Get final model
        final_model = copy.deepcopy(model.image_encoder)
        
        # Save models if requested
        if args.save:
            # Save best accuracy model
            acc_save_path = os.path.join(
                args.save, 
                f"{current_dataset}_best_accuracy_epoch{best_accuracy_epoch}.pt"
            )
            torch.save(best_accuracy_model.state_dict(), acc_save_path)
            print(f"Saved best accuracy model (val_acc: {best_val_accuracy:.2f}%, epoch: {best_accuracy_epoch}) to {acc_save_path}")
            
            # Save best FIM model
            fim_save_path = os.path.join(
                args.save, 
                f"{current_dataset}_best_fim_epoch{best_fim_epoch}.pt"
            )
            torch.save(best_fim_model.state_dict(), fim_save_path)
            print(f"Saved best FIM model (logtr: {best_fim_logtr:.4f}, epoch: {best_fim_epoch}) to {fim_save_path}")
            
            # Save final model
            final_save_path = os.path.join(
                args.save, 
                f"{current_dataset}_final_epoch{num_epochs}.pt"
            )
            torch.save(final_model.state_dict(), final_save_path)
            print(f"Saved final model (epoch: {num_epochs}) to {final_save_path}")
        
        # Find best metrics
        best_train_accuracy: float = max(result['train_accuracy'] for result in epoch_results)
        best_val_accuracy: float = max(result['validation_accuracy'] for result in epoch_results)
        best_fim_logtr: float = max(result['fim_logtr'] for result in epoch_results)
        lowest_loss: float = min(result['loss'] for result in epoch_results)
        
        # Store all results for this dataset
        results[current_dataset] = {
            'epoch_history': epoch_results,
            'best_metrics': {
                'accuracy': {
                    'train': best_train_accuracy,
                    'validation': best_val_accuracy,
                    'epoch': best_accuracy_epoch,
                    'model': best_accuracy_model,
                    'save_path': os.path.join(args.save, f"{current_dataset}_best_accuracy_epoch{best_accuracy_epoch}.pt") if args.save else None
                },
                'fim_logtr': {
                    'value': best_fim_logtr,
                    'epoch': best_fim_epoch,
                    'model': best_fim_model,
                    'save_path': os.path.join(args.save, f"{current_dataset}_best_fim_epoch{best_fim_epoch}.pt") if args.save else None
                },
                'loss': lowest_loss
            },
            'training_details': {
                'num_epochs': num_epochs,
                'learning_rate': args.lr,
                'weight_decay': args.wd,
                'batch_size': args.batch_size
            },
            'final_model': {
                'model': final_model,
                'epoch': num_epochs,
                'train_accuracy': epoch_results[-1]['train_accuracy'],
                'validation_accuracy': epoch_results[-1]['validation_accuracy'],
                'fim_logtr': epoch_results[-1]['fim_logtr'],
                'loss': epoch_results[-1]['loss'],
                'save_path': os.path.join(args.save, f"{current_dataset}_final_epoch{num_epochs}.pt") if args.save else None
            }
        }
    
    return results


def save_finetune_results(args: Namespace, results: Dict) -> None:
    results_dir = os.path.join(args.save, "tmp")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "finetune_results.json")
    
    # Convert models to paths and save them
    serializable_results = {}
    for dataset, dataset_results in results.items():
        # Create dataset-specific directory
        dataset_dir = os.path.join(results_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        serializable_results[dataset] = {
            'epoch_history': dataset_results['epoch_history'],
            'best_metrics': {
                'accuracy': {
                    'train': dataset_results['best_metrics']['accuracy']['train'],
                    'validation': dataset_results['best_metrics']['accuracy']['validation'],
                    'epoch': dataset_results['best_metrics']['accuracy']['epoch'],
                },
                'fim_logtr': {
                    'value': dataset_results['best_metrics']['fim_logtr']['value'],
                    'epoch': dataset_results['best_metrics']['fim_logtr']['epoch'],
                },
                'loss': dataset_results['best_metrics']['loss']
            },
            'training_details': dataset_results['training_details'],
            'final_model': {
                'epoch': dataset_results['final_model']['epoch'],
                'train_accuracy': dataset_results['final_model']['train_accuracy'],
                'validation_accuracy': dataset_results['final_model']['validation_accuracy'],
                'fim_logtr': dataset_results['final_model']['fim_logtr'],
                'loss': dataset_results['final_model']['loss'],
            }
        }
        
        # Save models and update paths in serializable results
        # Best accuracy model
        if dataset_results['best_metrics']['accuracy']['model'] is not None:
            acc_model_path = os.path.join(dataset_dir, f"best_accuracy_model.pt")
            torch.save(dataset_results['best_metrics']['accuracy']['model'].state_dict(), acc_model_path)
            serializable_results[dataset]['best_metrics']['accuracy']['model_path'] = acc_model_path
            
        # Best FIM model
        if dataset_results['best_metrics']['fim_logtr']['model'] is not None:
            fim_model_path = os.path.join(dataset_dir, f"best_fim_model.pt")
            torch.save(dataset_results['best_metrics']['fim_logtr']['model'].state_dict(), fim_model_path)
            serializable_results[dataset]['best_metrics']['fim_logtr']['model_path'] = fim_model_path
            
        # Final model
        if dataset_results['final_model']['model'] is not None:
            final_model_path = os.path.join(dataset_dir, f"final_model.pt")
            torch.save(dataset_results['final_model']['model'].state_dict(), final_model_path)
            serializable_results[dataset]['final_model']['model_path'] = final_model_path
    
    # Save the JSON with all results and model paths
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Saved fine-tuning results to {results_path}")
    print("Model files saved in dataset-specific directories under", results_dir)

def load_finetune_results(args: Namespace) -> Dict:
    results_path = os.path.join(args.save, "tmp", "finetune_results.json")
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No fine-tuning results found at {results_path}")
        
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load models from saved paths
    for dataset, dataset_results in results.items():
        # Load best accuracy model if path exists
        if 'model_path' in dataset_results['best_metrics']['accuracy']:
            model_path = dataset_results['best_metrics']['accuracy']['model_path']
            if os.path.exists(model_path):
                encoder = ImageEncoder(args)
                encoder.load_state_dict(torch.load(model_path))
                dataset_results['best_metrics']['accuracy']['model'] = encoder
            
        # Load best FIM model if path exists
        if 'model_path' in dataset_results['best_metrics']['fim_logtr']:
            model_path = dataset_results['best_metrics']['fim_logtr']['model_path']
            if os.path.exists(model_path):
                encoder = ImageEncoder(args)
                encoder.load_state_dict(torch.load(model_path))
                dataset_results['best_metrics']['fim_logtr']['model'] = encoder
            
        # Load final model if path exists
        if 'model_path' in dataset_results['final_model']:
            model_path = dataset_results['final_model']['model_path']
            if os.path.exists(model_path):
                encoder = ImageEncoder(args)
                encoder.load_state_dict(torch.load(model_path))
                dataset_results['final_model']['model'] = encoder
    
    return results


def main() -> None:
    from args import parse_arguments
    args: Namespace = parse_arguments()
    print(args)
    
    # Save pretrained model first

    encoder: ImageEncoder = ImageEncoder(args)
    torch.save(encoder, os.path.join(args.save, "pretrained.pt"))
    
    # Dictionary to store single task results
    single_task_results = {}
    
    # List of all datasets to process
    if not args.train_dataset:
        datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    else:
        datasets = args.train_dataset
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}")
        
        # Fine-tune the model
        results = finetune_model(
            args=args,
        )
        
        # Store test accuracies for normalization in task addition
        dataset_results = results[dataset_name]
        single_task_results[dataset_name] = {
            "test_acc": dataset_results["best_metrics"]["accuracy"]["value"],
            "best_model_path": dataset_results["best_metrics"]["accuracy"]["save_path"],
            "best_fim_path": dataset_results["best_metrics"]["fim_logtr"]["save_path"],
            "final_model_path": dataset_results["final_model"]["save_path"]
        }
        
        # Save the best model as the finetuned model for task vectors
        # best_model = dataset_results["best_metrics"]["accuracy"]["model"]
        # finetuned_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
        # torch.save(best_model, finetuned_path)
    
    # Save single task results for normalization
    single_task_path = os.path.join(args.save, "single_task_results.json")
    # with open(single_task_path, 'w') as f:
    #     json.dump(single_task_results, f, indent=4)
    
    print("\nSaved all models and results:")
    print(f"1. Pretrained model: {os.path.join(args.save, 'pretrained.pt')}")
    print(f"2. Single task results: {single_task_path}")
    print("3. For each dataset:")
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        print(f"  - Finetuned model: {os.path.join(args.save, f'{dataset_name}_finetuned.pt')}")
        print(f"  - Best accuracy model: {single_task_results[dataset_name]['best_model_path']}")
        print(f"  - Best FIM model: {single_task_results[dataset_name]['best_fim_path']}")
        print(f"  - Final model: {single_task_results[dataset_name]['final_model_path']}")

if __name__ == "__main__":
    main()