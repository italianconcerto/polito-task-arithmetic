# Results Schema

## Finetuning Results

Each dataset (e.g. "MNIST", "SVHN", etc.) contains:

### Epoch History
Array of objects with metrics for each epoch:
- `epoch`: int
- `loss`: float 
- `train_accuracy`: float
- `validation_loss`: float
- `validation_accuracy`: float
- `fim_logtr`: float

### Best Metrics
Best metrics achieved during training:

**Accuracy**
- `train`: float
- `validation`: float
- `epoch`: int
- `model`: ImageEncoder (in memory)
- `model_path`: str (when serialized)

**FIM Log Trace**
- `value`: float
- `epoch`: int
- `model`: ImageEncoder (in memory)
- `model_path`: str (when serialized)

**Loss**
- `loss`: float

### Training Details
- `num_epochs`: int
- `learning_rate`: float
- `weight_decay`: float
- `batch_size`: int

### Final Model
- `model`: ImageEncoder (in memory)
- `model_path`: str (when serialized)
- `epoch`: int
- `train_accuracy`: float
- `validation_accuracy`: float
- `fim_logtr`: float
- `loss`: float

## Evaluation Results

Each dataset contains results for different model types ("best_accuracy", "best_fim", "final"):

### Model Info
- `type`: str
- `path`: str

### Train/Validation/Test Metrics
Each split contains:
- `accuracy`: float
- `loss`: float
- `num_samples`: int

### Additional Metrics
- `fim_logtr`: float

## Task Addition Results

### Best Alpha
- `best_alpha`: float

### Alpha Results
For each alpha value:
- `alpha`: float
- Dataset results for each dataset:
  - Train/validation/test metrics:
    - `accuracy`: float
    - `loss`: float
  - `normalized_acc`: float
  - `single_task_acc`: float
  - `fim_logtr`: float
- Average metrics:
  - `normalized_acc`: float
  - `absolute_acc`: float

### Best Results
Same structure as alpha results for the best performing alpha

## Complete Experiment Results
- `finetuning`: Finetuning results
- `evaluation`: Evaluation results  
- `task_addition`: Task addition results

## JSON Schema

# 
finetuning_results = {
    "dataset_name": {  # e.g., "MNIST", "SVHN", etc.
        "epoch_history": [
            {
                "epoch": int,
                "loss": float,
                "train_accuracy": float,
                "validation_loss": float,
                "validation_accuracy": float,
                "fim_logtr": float
            }
        ],
        "best_metrics": {
            "accuracy": {
                "train": float,
                "validation": float,
                "epoch": int,
                "model": ImageEncoder,  # In memory
                "model_path": str  # When serialized
            },
            "fim_logtr": {
                "value": float,
                "epoch": int,
                "model": ImageEncoder,  # In memory
                "model_path": str  # When serialized
            },
            "loss": float
        },
        "training_details": {
            "num_epochs": int,
            "learning_rate": float,
            "weight_decay": float,
            "batch_size": int
        },
        "final_model": {
            "model": ImageEncoder,  # In memory
            "model_path": str,  # When serialized
            "epoch": int,
            "train_accuracy": float,
            "validation_accuracy": float,
            "fim_logtr": float,
            "loss": float
        }
    }
}

# Evaluation Results Schema
evaluation_results = {
    "dataset_name": {  # e.g., "MNIST", "SVHN", etc.
        "model_type": {  # "best_accuracy", "best_fim", "final"
            "model_info": {
                "type": str,
                "path": str
            },
            "train": {
                "accuracy": float,
                "loss": float,
                "num_samples": int
            },
            "validation": {
                "accuracy": float,
                "loss": float,
                "num_samples": int
            },
            "test": {
                "accuracy": float,
                "loss": float,
                "num_samples": int
            },
            "fim_logtr": float
        }
    }
}

# Task Addition Results Schema
task_addition_results = {
    "best_alpha": float,
    "alpha_results": {
        float: {  # alpha value as key
            "alpha": float,
            "dataset_results": {
                "dataset_name": {
                    "train": {
                        "accuracy": float,
                        "loss": float
                    },
                    "validation": {
                        "accuracy": float,
                        "loss": float
                    },
                    "test": {
                        "accuracy": float,
                        "loss": float
                    },
                    "normalized_acc": float,
                    "single_task_acc": float,
                    "fim_logtr": float
                }
            },
            "average_metrics": {
                "normalized_acc": float,
                "absolute_acc": float
            }
        }
    },
    "best_results": dict  # Same structure as alpha_results[alpha]
}

# Complete Experiment Results Schema
experiment_results = {
    "finetuning": finetuning_results,
    "evaluation": evaluation_results,
    "task_addition": task_addition_results
}