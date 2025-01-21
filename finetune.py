import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_save

def get_balanced_sampler(dataset):
    targets = []
    for _, label in dataset:
        if isinstance(label, dict):
            label = label['labels']
        targets.append(label)
    
    targets = torch.tensor(targets)
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def train_one_epoch(model, train_loader, optimizer, criterion, args):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        batch = maybe_dictionarize(batch)
        x, y = batch['images'].to(args.device), batch['labels'].to(args.device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
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

def main():
    args = parse_arguments()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
    
    epochs_mapping = {
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SVHN": 4
    }
    
    for dataset_name in epochs_mapping:
        print(f"\nFine-tuning on {dataset_name}")
        
        # Initialize model
        encoder = ImageEncoder(args)
        head = get_classification_head(args, f"{dataset_name}Val")
        model = ImageClassifier(encoder, head)
        model.freeze_head()
        model = model.to(args.device)
        
        # Setup optimizer and loss
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.wd
        )
        criterion = nn.CrossEntropyLoss()
        
        # Get dataset
        dataset = get_dataset(
            f"{dataset_name}Val",
            preprocess=model.train_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=2
        )
        if args.balanced_sampler:
            # Create balanced sampler
            sampler = get_balanced_sampler(dataset.train_dataset)
            
            # Create dataloader with balanced sampler
            train_loader = torch.utils.data.DataLoader(
                dataset.train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=2
            )
        else:
            train_loader = get_dataloader(dataset, is_train=True, args=args)
        
        # Training loop
        for epoch in range(epochs_mapping[dataset_name]):
            print(f"\nEpoch {epoch + 1}/{epochs_mapping[dataset_name]}")
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, args
            )
            
            print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Save the full encoder model
        save_path = f"{args.save}/{dataset_name}_finetuned.pt"
        torch.save(model.image_encoder, save_path)
        print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()