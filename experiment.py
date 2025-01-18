from typing import Dict, Optional
import os
from argparse import Namespace
from finetune import finetune_model
from eval_single_task import evaluate_models
from args import parse_arguments

def run_experiment(args: Namespace) -> Dict:
    """
    Run the complete experiment pipeline: finetuning followed by evaluation.
    """
    print("\n=== Starting Experiment ===")
    
    # Create necessary directories
    if not args.save:
        os.makedirs(args.save, exist_ok=True)
    
    # Step 1: Fine-tuning
    print("\n=== Starting Fine-tuning Phase ===")
    finetuning_results = finetune_model(args)
    
    # Step 2: Evaluation
    print("\n=== Starting Evaluation Phase ===")
    evaluation_results = evaluate_models(args, finetuning_results)
    
    # Combine results
    experiment_results = {
        'finetuning': finetuning_results,
        'evaluation': evaluation_results
    }
    
    print("\n=== Experiment Completed ===")
    return experiment_results

def main() -> None:
    # Parse arguments
    args = parse_arguments()
    print("Running experiment with args:", args)
    
    # Run experiment
    results = run_experiment(args)
    
    print("\nExperiment completed successfully!")
    print(f"Results saved in: {args.save}")

if __name__ == "__main__":
    main()