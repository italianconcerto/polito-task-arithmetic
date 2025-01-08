import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_save

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
        train_loader = get_dataloader(dataset, is_train=True, args=args)
        
        # Training loop
        for epoch in range(epochs_mapping[dataset_name]):
            print(f"\nEpoch {epoch + 1}/{epochs_mapping[dataset_name]}")
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, args
            )
            
            print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Save the final model's state dict
        save_path = f"{args.save}/{dataset_name}_finetuned.pt"
        torch_save(model.image_encoder, save_path)
        print(f"Saved final checkpoint to {save_path}")

if __name__ == "__main__":
    main()