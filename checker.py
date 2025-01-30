import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm
from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import torch_save
from collections import Counter
import matplotlib.pyplot as plt
from finetune import get_balanced_sampler



def check_sampler(train_loader, num_batches_to_check=5):
    print(f"Checking first {num_batches_to_check} batches:")
    
    all_indices = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches_to_check:
            break
            
        # batch = maybe_dictionarize(batch)
        labels = batch['labels']
        
        # If your sampler has indices attribute (like DistributedSampler)
        if hasattr(train_loader.sampler, 'indices'):
            batch_indices = train_loader.sampler.indices[
                batch_idx * train_loader.batch_size:
                (batch_idx + 1) * train_loader.batch_size
            ]
            all_indices.extend(batch_indices)
        
        all_labels.extend(labels.cpu().numpy())
        
        print(f"\nBatch {batch_idx}:")
        print(f"Labels in batch: {labels.cpu().numpy()}")
        print(f"Batch size: {len(labels)}")
        
        if hasattr(train_loader.sampler, 'indices'):
            print(f"Indices in batch: {batch_indices}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total samples checked: {len(all_labels)}")
    print(f"Unique labels: {np.unique(all_labels)}")
    print(f"Label distribution:\n{Counter(all_labels)}")
    
    if all_indices:
        print(f"Number of unique indices: {len(set(all_indices))}")
        print(f"Are indices sequential? {all_indices == sorted(all_indices)}")




def additional_sampler_checks(train_loader):
    sampler = train_loader.sampler
    
    print("\nSampler Information:")
    print(f"Sampler type: {type(sampler)}")
    
    # For DistributedSampler
    if hasattr(sampler, 'num_replicas'):
        print(f"Number of replicas: {sampler.num_replicas}")
        print(f"Rank: {sampler.rank}")
        print(f"Shuffle: {sampler.shuffle}")
    
    # For WeightedRandomSampler
    if hasattr(sampler, 'weights'):
        weights = sampler.weights.cpu().numpy()
        print(f"Weight distribution: min={weights.min():.4f}, max={weights.max():.4f}")
        print(f"Unique weights: {np.unique(weights)}")
    
    # Check dataset size vs sampler size
    if hasattr(sampler, '__len__'):
        print(f"Sampler length: {len(sampler)}")
        print(f"Dataset length: {len(train_loader.dataset)}")
        print(f"Are they equal? {len(sampler) == len(train_loader.dataset)}")

def plot_distribution(counts, title):
    plt.figure(figsize=(10, 5))
    classes = list(range(len(counts)))
    plt.bar(classes, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(classes)
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_dataset_balance(dataset_name, args, num_batches=500):
    # Set required args if not present
    if not hasattr(args, 'model'):
        args.model = 'ViT-B-32'
    if not hasattr(args, 'pretrained'):
        args.pretrained = 'openai'
    if not hasattr(args, 'device'):
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model (needed for preprocessing)
    encoder = ImageEncoder(args)
    head = get_classification_head(args, f"{dataset_name}Val")
    model = ImageClassifier(encoder, head)
    
    # Get dataset
    dataset = get_dataset(
        f"{dataset_name}Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Analyze original dataset distribution
    original_labels = []
    for _, label in dataset.train_dataset:
        if isinstance(label, dict):
            label = label['labels']
        original_labels.append(label)
    
    original_counts = torch.bincount(torch.tensor(original_labels))
    print(f"\n{dataset_name} Original Dataset Distribution:")
    print("Total samples:", len(original_labels))
    print("Class counts:", original_counts.tolist())
    print("Min count:", original_counts.min().item())
    print("Max count:", original_counts.max().item())
    print("Imbalance ratio:", (original_counts.max() / original_counts.min()).item())
    
    # Create and analyze balanced sampler
    sampler = get_balanced_sampler(dataset.train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2
    )
    
    # Collect labels from balanced batches
    balanced_labels = []
    total_samples = 0
    print("\nCollecting samples from balanced loader...")
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        if isinstance(batch, dict):  # Handle dictionary-style batches
            labels = batch['labels']
        else:
            _, labels = batch
        balanced_labels.extend(labels.cpu().numpy())
        total_samples += len(labels)
        if i % 50 == 0:
            print(f"Processed {i} batches, {total_samples} samples...")
    
    balanced_counts = torch.bincount(torch.tensor(balanced_labels))
    print(f"\nBalanced Sampler Distribution (after {num_batches} batches):")
    print("Total samples:", len(balanced_labels))
    print("Class counts:", balanced_counts.tolist())
    print("Min count:", balanced_counts.min().item())
    print("Max count:", balanced_counts.max().item())
    print("Imbalance ratio:", (balanced_counts.max() / balanced_counts.min()).item())
    
    # Plot distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plot_distribution(original_counts, f'{dataset_name} Original Distribution')
    
    plt.subplot(1, 2, 2)
    plot_distribution(balanced_counts, f'{dataset_name} Balanced Distribution')
    
    plt.tight_layout()
    plt.show()

    # After creating sampler
    print(f"\nBalanced Sampler Statistics:")
    print(f"Samples per class: {original_counts.min().item()}")
    print(f"Total samples per epoch: {original_counts.min().item() * len(original_counts)}")
    
    # Collect labels from FULL epoch
    balanced_labels = []
    for batch in train_loader:
        if isinstance(batch, dict):
            labels = batch['labels']
        else:
            _, labels = batch
        balanced_labels.extend(labels.cpu().numpy())
        
    balanced_counts = torch.bincount(torch.tensor(balanced_labels))
    print(f"\nBalanced Sampler Distribution (after full epoch):")
    print("Total samples:", len(balanced_labels))
    print("Class counts:", balanced_counts.tolist())
    print("Min count:", balanced_counts.min().item())
    print("Max count:", balanced_counts.max().item())
    print("Imbalance ratio:", (balanced_counts.max() / balanced_counts.min()).item())

def main():
    args = parse_arguments()
    
    # List of datasets to check
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Analyzing {dataset_name}")
        print(f"{'='*50}")
        analyze_dataset_balance(dataset_name, args)

if __name__ == "__main__":
    main()

