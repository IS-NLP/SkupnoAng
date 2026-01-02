from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import cos_sim

from losses import trainer_cl_3classes, trainer_mnr, trainer_cl
from dataprocessing.binaryset import get_data
from Evaluation.inbuilt import get_bin_eval, get_ret_eval
from Evaluation.recalleval import MyRecallEval
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss, TripletLoss

def main(args, hyperparameter_search=False, train_dataset=None, valid_dataset=None, test_dataset=None):
    if train_dataset is None:
        train_dataset, valid_dataset, test_dataset = get_data(valid=False)
    print(":)")
    model_name = "models\jina-embeddings-v2-small-en" 
    base_model = SentenceTransformer(model_name)

    print("meow")
    loss = ContrastiveLoss(base_model)

    evaluator = get_ret_eval(valid_dataset)

    trainer = SentenceTransformerTrainer(
        model = base_model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=loss,
        evaluator=evaluator,
        args=args
    )
    print("woof")
    #trainer = trainer_mnr(fine_model, train_dataset, valid_dataset, args)
    trainer.train()
    print("moo")


    if hyperparameter_search:      
        ev = MyRecallEval(test_dataset)
        metrics = ev(trainer.model)
        print(metrics)
        return metrics['eval_recall@10']
    ev = MyRecallEval(test_dataset)
    metrics = ev(trainer.model)
    print(metrics)
    
    model = trainer.model
    # Test with clear examples
    test_cases = [
        ("It is raining", "It is not raining", 1),  # Contradict - should be CLOSE
        ("It is raining", "The weather is wet", 0),  # Confirm - should be FAR
    ]

    for s1, s2, expected in test_cases:
        emb1 = model.encode(s1)
        emb2 = model.encode(s2)
        sim = cos_sim(emb1, emb2).item()
        print(f"'{s1}' vs '{s2}'")
        print(f"  Similarity: {sim:.3f} (Expected: {'HIGH' if expected==1 else 'LOW'})")
        print()

if __name__ == "__main__":
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/tuned_model",
        # Optional training parameters:
        num_train_epochs=1.0,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1.2247359733257542e-05,
        seed=42,
        metric_for_best_model=f"recall@10",
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
        eval_steps=50, # how often we eval
        #save_strategy="best",
        torch_empty_cache_steps = None,
        save_steps=50,
        save_total_limit=2,
        max_grad_norm= 0.8774817671930895,
        logging_steps=100,
        #run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )
    main(args, False)
  
