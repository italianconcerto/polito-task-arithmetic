from typing import Dict, Optional
import os
from argparse import Namespace
from finetune import finetune_model
from eval_single_task import evaluate_models
from eval_task_addition import evaluate_multitask_model
from args import parse_arguments
from task_vectors import NonLinearTaskVector
import pandas as pd
import numpy as np

def save_results_table(args: Namespace, results: Dict) -> None:
    """
    Save results in a CSV format matching the paper's table structure.
    """
    datasets = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"]
    
    # Initialize DataFrame with MultiIndex
    index = pd.MultiIndex.from_product([
        ['Single-task Acc.', 'Single-task Acc.', 'log Tr[F̂]',
         'Single-task Acc.', 'Single-task Acc.', 'log Tr[F̂]',
         'Single-task Acc.', 'Single-task Acc.', 'log Tr[F̂]',
         'Absolute Acc.', 'Normalized Acc.', 'Absolute Acc.', 'Normalized Acc.', 'log Tr[F̂]'],
        ['Test', 'Train', 'Train', 'Test', 'Train', 'Train', 'Test', 'Train', 'Train',
         'Test', 'Test', 'Train', 'Train', 'Train']
    ])
    
    df = pd.DataFrame(index=index, columns=datasets + ['Average'])
    
    # Fill pre-trained model metrics
    pretrained_results = results['evaluation']['pretrained']
    for dataset in datasets:
        df.loc[('Single-task Acc.', 'Test'), dataset] = pretrained_results[dataset]['test']['accuracy']
        df.loc[('Single-task Acc.', 'Train'), dataset] = pretrained_results[dataset]['train']['accuracy']
        df.loc[('log Tr[F̂]', 'Train'), dataset] = pretrained_results[dataset]['fim_logtr']
    
    # Fill metrics before scaling & addition (fine-tuned models)
    finetuned_results = results['evaluation']['finetuned']
    for dataset in datasets:
        df.loc[('Single-task Acc.', 'Test'), dataset] = finetuned_results[dataset]['test']['accuracy']
        df.loc[('Single-task Acc.', 'Train'), dataset] = finetuned_results[dataset]['train']['accuracy']
        df.loc[('log Tr[F̂]', 'Train'), dataset] = finetuned_results[dataset]['fim_logtr']
    
    # Fill metrics after scaling
    scaled_results = results['evaluation']['scaled']
    for dataset in datasets:
        df.loc[('Single-task Acc.', 'Test'), dataset] = scaled_results[dataset]['test']['accuracy']
        df.loc[('Single-task Acc.', 'Train'), dataset] = scaled_results[dataset]['train']['accuracy']
        df.loc[('log Tr[F̂]', 'Train'), dataset] = scaled_results[dataset]['fim_logtr']
    
    # Fill metrics after addition
    task_addition = results['task_addition']['best_results']
    for dataset in datasets:
        dataset_results = task_addition['dataset_results'][dataset]
        df.loc[('Absolute Acc.', 'Test'), dataset] = dataset_results['test']['accuracy']
        df.loc[('Normalized Acc.', 'Test'), dataset] = dataset_results['normalized_acc']
        df.loc[('Absolute Acc.', 'Train'), dataset] = dataset_results['validation']['accuracy']
        df.loc[('Normalized Acc.', 'Train'), dataset] = dataset_results['validation']['accuracy'] / dataset_results['single_task_acc']
        df.loc[('log Tr[F̂]', 'Train'), dataset] = dataset_results.get('fim_logtr', np.nan)
    
    # Calculate averages
    df['Average'] = df[datasets].mean(axis=1)
    
    # Save to CSV
    csv_path = os.path.join(args.save, "results_table.csv")
    df.to_csv(csv_path)
    print(f"\nSaved results table to: {csv_path}")
    
    # Also save as a formatted text file for easy viewing
    txt_path = os.path.join(args.save, "results_table.txt")
    with open(txt_path, 'w') as f:
        f.write(df.to_string())
    print(f"Saved formatted table to: {txt_path}")

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
    
    # Save results table
    save_results_table(args, experiment_results)
    
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


    # reference = pd.read_csv("reference_table.csv", index_col=[0,1])
    # results = pd.read_csv("results_table.csv", index_col=[0,1])

    # # Compare results
    # comparison = pd.DataFrame({
    #     'Reference': reference['Average'],
    #     'Our Results': results['Average'],
    #     'Difference': results['Average'] - reference['Average']
    # })
    # print(comparison)
if __name__ == "__main__":
    main()