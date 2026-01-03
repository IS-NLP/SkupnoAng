from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from losses import trainer_cl, trainer_mnr, trainer3
from dataprocessing.triplets_set import get_data, triple
from Evaluation.inbuilt import get_bin_eval, get_ret_eval
from Evaluation.recalleval import MyRecallEval

def main3():
    train_dataset, valid_dataset, test_dataset = get_data()
    print(":)")
    model_name = "models\jina-embeddings-v2-small-en" 
    base_model = SentenceTransformer(model_name, device="cuda")

    fine_model = base_model

    k = 10
    print("quack")
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/tuned_model3",
        # Optional training parameters:
        num_train_epochs=0.5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1.2247359733257542e-05,
        seed=42,
        metric_for_best_model=f"eval_normalized_recall@10",
        #greater_is_better=False,
        load_best_model_at_end=True,
        weight_decay=0.09092585204374326,
        warmup_ratio=0.05503071687326718,

        #warmup_ratio=0.1,
        #fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        #bf16=False,  # Set to True if you have a GPU that supports BF16
        #batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100, # how often we eval
        #save_strategy="best",
        torch_empty_cache_steps = None,
        save_steps=100,
        save_total_limit=2,
        max_grad_norm= 0.8774817671930895,
        logging_steps=100,
        #run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )
    print("meow")
    trainer = trainer3(fine_model, train_dataset, valid_dataset, args)

    trainer.train()
    print("moo")
    trainer.save_model("meow3")
    ev = MyRecallEval(test_dataset)
    metrics = ev(trainer.model)
    print(metrics)
    return metrics

if __name__ == "__main__":
    main3()