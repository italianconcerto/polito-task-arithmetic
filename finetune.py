from typing import Dict, List, Optional, Tuple, Union, Literal
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

def train_one_epoch(
    model: ImageClassifier,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args: Namespace
) -> Tuple[float, float]:
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
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def finetune_model(
    args: Namespace,
    dataset_name: Optional[str] = None,
    custom_epochs: Optional[int] = None,
    save: bool = False
) -> Dict[str, Dict]:
    default_epochs_mapping: Dict[str, int] = {
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SVHN": 4
    }
    
    datasets_to_process: List[str] = [dataset_name] if dataset_name else list(default_epochs_mapping.keys())
    
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
        
        num_epochs: int = custom_epochs if custom_epochs else default_epochs_mapping[current_dataset]
        
        epoch_results: List[Dict[str, Union[int, float]]] = []
        
        # Track best models and their metrics
        best_accuracy: float = float('-inf')
        best_fim_logtr: float = float('-inf')
        best_accuracy_model: Optional[ImageEncoder] = None
        best_fim_model: Optional[ImageEncoder] = None
        best_accuracy_epoch: int = 0
        best_fim_epoch: int = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training step
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, args
            )
            
            # Calculate FIM log trace
            fim_logtr: float = train_diag_fim_logtr(args, model, current_dataset)
            
            # Update best models if needed
            if train_acc > best_accuracy:
                best_accuracy = train_acc
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
                'accuracy': train_acc,
                'fim_logtr': fim_logtr,
            }
            epoch_results.append(epoch_data)
            
            print(
                f"Training Loss: {train_loss:.4f}, "
                f"Accuracy: {train_acc:.2f}%, "
                f"FIM Log Trace: {fim_logtr:.4f}"
            )
        
        # Get final model
        final_model = copy.deepcopy(model.image_encoder)
        
        # Save models if requested
        if save and args.save:
            # Save best accuracy model
            acc_save_path = os.path.join(
                args.save, 
                f"{current_dataset}_best_accuracy_epoch{best_accuracy_epoch}.pt"
            )
            torch.save(best_accuracy_model.state_dict(), acc_save_path)
            print(f"Saved best accuracy model (acc: {best_accuracy:.2f}%, epoch: {best_accuracy_epoch}) to {acc_save_path}")
            
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
        best_accuracy: float = max(result['accuracy'] for result in epoch_results)
        best_fim_logtr: float = max(result['fim_logtr'] for result in epoch_results)
        lowest_loss: float = min(result['loss'] for result in epoch_results)
        
        # Store all results for this dataset
        results[current_dataset] = {
            'epoch_history': epoch_results,
            'best_metrics': {
                'accuracy': {
                    'value': best_accuracy,
                    'epoch': best_accuracy_epoch,
                    'model': best_accuracy_model,
                    'save_path': os.path.join(args.save, f"{current_dataset}_best_accuracy_epoch{best_accuracy_epoch}.pt") if save and args.save else None
                },
                'fim_logtr': {
                    'value': best_fim_logtr,
                    'epoch': best_fim_epoch,
                    'model': best_fim_model,
                    'save_path': os.path.join(args.save, f"{current_dataset}_best_fim_epoch{best_fim_epoch}.pt") if save and args.save else None
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
                'accuracy': epoch_results[-1]['accuracy'],
                'fim_logtr': epoch_results[-1]['fim_logtr'],
                'loss': epoch_results[-1]['loss'],
                'save_path': os.path.join(args.save, f"{current_dataset}_final_epoch{num_epochs}.pt") if save and args.save else None
            }
        }
    
    return results

def main() -> None:
    from args import parse_arguments
    args: Namespace = parse_arguments()
    print(args)
    
    results = finetune_model(
        args=args,
        dataset_name="DTD",
    )
    
    # Example of accessing results
    dtd_results = results["DTD"]
    print(f"Best accuracy: {dtd_results['best_metrics']['accuracy']['value']:.2f}% (epoch {dtd_results['best_metrics']['accuracy']['epoch']})")
    print(f"Best FIM log trace: {dtd_results['best_metrics']['fim_logtr']['value']:.4f} (epoch {dtd_results['best_metrics']['fim_logtr']['epoch']})")
    
    print(f"All the results: {results}")

if __name__ == "__main__":
    main()