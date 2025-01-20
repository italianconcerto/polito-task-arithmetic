import json
import random
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import train_diag_fim_logtr

def evaluate(model, data_loader, args):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating')
        for batch in progress_bar:
            batch = maybe_dictionarize(batch)
            x, y = batch['images'].to(args.device), batch['labels'].to(args.device)
            
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            progress_bar.set_postfix({
                'acc': 100. * correct / total
            })
    
    return 100. * correct / total

def evaluate_single_task(args, dataset_name):
    # Initialize model
    encoder = ImageEncoder(args)
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    
    # Load fine-tuned weights
    checkpoint_path = f"{args.save}/{dataset_name}_finetuned.pt"
    loaded_encoder = torch.load(checkpoint_path, map_location=args.device)
    model.image_encoder = loaded_encoder  # Replace encoder with loaded model
    model = model.to(args.device)
    
    results = {}
    
    train_dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )
    train_loader = get_dataloader(train_dataset, is_train=True, args=args)
    train_acc = evaluate(model, train_loader, args)
    results['train_acc'] = train_acc
    
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
    results['val_acc'] = val_acc
    
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
    results['test_acc'] = test_acc
    
    # Compute Fisher Information Matrix log-trace
    fim_logtr = train_diag_fim_logtr(args, model, dataset_name+"Val")
    results['fim_logtr'] = fim_logtr
    
    print(f"\nResults for {dataset_name}:")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"FIM Log-Trace: {fim_logtr:.4f}")
    
    return results

def main():
    args = parse_arguments()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
    
    datasets = args.eval_datasets
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\nEvaluating {dataset_name}")
        results = evaluate_single_task(args, dataset_name)
        all_results[dataset_name] = results
    
    # Save all results
    save_path = f"{args.save}/single_task_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved results to {save_path}")

if __name__ == "__main__":
    main()