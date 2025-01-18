from typing import Dict, Optional
import os
from argparse import Namespace
from finetune import finetune_model
from eval_single_task import evaluate_models
from eval_task_addition import evaluate_multitask_model
from args import parse_arguments

def run_experiment(args: Namespace) -> Dict:
    """
    Run the complete experiment pipeline: finetuning, evaluation, and task addition.
    """
    print("\n=== Starting Experiment ===")
    
    # Create necessary directories
    os.makedirs(args.save, exist_ok=True)
    
    # Step 1: Fine-tuning
    print("\n=== Starting Fine-tuning Phase ===")
    finetuning_results = finetune_model(args)
    
    # Step 2: Evaluation
    print("\n=== Starting Evaluation Phase ===")
    evaluation_results = evaluate_models(args, finetuning_results)
    
    # Step 3: Task Addition
    print("\n=== Starting Task Addition Phase ===")
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    pretrained_path = os.path.join(args.save, "pretrained.pt")
    
    print("Building task vectors...")
    task_vectors = {}
    for dataset_name in datasets:
        finetuned_path = os.path.join(args.save, f"{dataset_name}_finetuned.pt")
        task_vectors[dataset_name] = NonLinearTaskVector(
            pretrained_checkpoint=pretrained_path,
            finetuned_checkpoint=finetuned_path
        )
    
    # Find optimal alpha
    print("\nFinding optimal alpha...")
    alphas = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 in 0.05 steps
    best_alpha = 0
    best_avg_normalized_acc = 0
    best_results = None
    
    alpha_results = {}
    for alpha in alphas:
        results = evaluate_multitask_model(
            args=args,
            datasets=datasets,
            task_vectors=task_vectors,
            alpha=alpha,
            pretrained_path=pretrained_path,
            save=True
        )
        alpha_results[alpha] = results
        avg_normalized_acc = results['average_metrics']['normalized_acc']
        
        if avg_normalized_acc > best_avg_normalized_acc:
            best_avg_normalized_acc = avg_normalized_acc
            best_alpha = alpha
            best_results = results
    
    task_addition_results = {
        'best_alpha': best_alpha,
        'alpha_results': alpha_results,
        'best_results': best_results
    }
    
    # Combine all results
    experiment_results = {
        'finetuning': finetuning_results,
        'evaluation': evaluation_results,
        'task_addition': task_addition_results
    }
    
    print("\n=== Experiment Completed ===")
    return experiment_results

def main() -> None:
    # Parse arguments
    args = parse_arguments()
    print("Running experiment with args:", args)
    
    # Run experiment
    results = run_experiment(args)
    
    # Print final summary
    print("\nExperiment completed successfully!")
    print(f"Results saved in: {args.save}")
    print(f"\nTask Addition Results:")
    print(f"Best alpha: {results['task_addition']['best_alpha']:.2f}")
    print(f"Best average normalized accuracy: {results['task_addition']['best_results']['average_metrics']['normalized_acc']:.4f}")
    print(f"Best average absolute accuracy: {results['task_addition']['best_results']['average_metrics']['absolute_acc']:.2f}%")

if __name__ == "__main__":
    main()