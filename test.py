from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from losses import trainer_cl, trainer_mnr, trainer3
from dataprocessing.triplets_set import get_data, triple
from Evaluation.inbuilt import get_bin_eval, get_ret_eval
from Evaluation.recalleval import MyRecallEval

def main3():
    train_dataset, valid_dataset, test_dataset = get_data()
    print(":)")
    model_name = "models/trained/model_mnlr2_test"
    model = SentenceTransformer(model_name)

    ev = MyRecallEval(test_dataset)
    metrics = ev(model)
    print(metrics)
    return metrics

if __name__ == "__main__":
    main3()