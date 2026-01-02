from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

from dataprocessing.binaryset import get_data
from Evaluation.recalleval import MyRecallEval

from losses import trainer_cl

# 1. Load the dataset
train_dataset, hyper_valid_dataset, test_dataset = get_data()


# 2. Create an evaluator to perform useful HPO
dev_evaluator = MyRecallEval(hyper_valid_dataset, cluster_min_hits=2, name="sts-dev", main_similarity=SimilarityFunction.COSINE)

# 3. Define the Hyperparameter Search Space
def hpo_search_space(trial):
    search_space = {
    'learning_rate': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5],
    'weight_decay': [0.01, 0.05, 0.1, 0.2],
    'warmup_ratio': [0.05, 0.1, 0.2],
    'max_grad_norm': [0.5, 1.0, 2.0]
    }
    return {
        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 0., 0.2),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.3),
        'max_grad_norm': trial.suggest_float("max_grad_norm", 0.5, 2.0, log=True),
    }

# 4. Define the Model Initialization
def hpo_model_init(trial):
    return SentenceTransformer("models\jina-embeddings-v2-small-en", device="cuda" )

# 5. Define the Loss Initialization
def hpo_loss_init(model):
    return losses.ContrastiveLoss(model)

# 6. Define the Objective Function
def hpo_compute_objective(metrics):
    print(metrics)
    return metrics["eval_sts-dev_normalized_recall@10"]

# 7. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    num_train_epochs=1.0,
    per_device_train_batch_size=16,
    seed=42,
    metric_for_best_model=f"recall@10",
    #greater_is_better=False,
    #warmup_ratio=0.1,
    #fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    #bf16=False,  # Set to True if you have a GPU that supports BF16
    #batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    #save_strategy="best",

    # Required parameter:
    output_dir="checkpoints",
    # Optional training parameters:
    # max_steps=10000, # We might want to limit the number of steps for HPO
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="no", # We don't need to evaluate/save during HPO
    save_strategy="no",
    logging_steps=40,
    run_name="hpo",  # Will be used in W&B if `wandb` is installed
)

# 8. Create the trainer with model_init rather than model
trainer = SentenceTransformerTrainer(
    model=None,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=hyper_valid_dataset,
    evaluator=dev_evaluator,
    model_init=hpo_model_init,
    loss=hpo_loss_init,
)

# 9. Perform the HPO
best_trial = trainer.hyperparameter_search(
    hp_space=hpo_search_space,
    compute_objective=hpo_compute_objective,
    n_trials=20,
    direction="maximize",
    backend="optuna",
)
print(best_trial)