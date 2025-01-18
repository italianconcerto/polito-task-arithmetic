
from argparse import Namespace
import json
import os

import torch

from finetune import finetune_model
from modeling import ImageEncoder


def experiment(): 
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
        
        
        
if __name__ == "__main__":
    experiment()