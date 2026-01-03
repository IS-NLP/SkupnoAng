from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from Evaluation.inbuilt import get_bin_eval, get_ret_eval
from Evaluation.recalleval import MyRecallEval

from dataprocessing.triplets_set import triple, ds_abs
from datasets import Dataset

def trainer_cl(m, train_dataset, valid_dataset, args):
    loss = ContrastiveLoss(m)

    if valid_dataset is not None:
        evaluator = get_ret_eval(valid_dataset)

    trainer = SentenceTransformerTrainer(
        model = m,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=loss,
        evaluator=evaluator,
        args=args
    )

    return trainer

def trainer_mnlr(m, train_dataset, valid_dataset, args):
    loss = MultipleNegativesRankingLoss(m)

    if valid_dataset is not None:
        evaluator = get_ret_eval(valid_dataset)

    trainer = SentenceTransformerTrainer(
        model = m,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=loss,
        evaluator=evaluator,
        args=args
    )
    return trainer

def trainer_mnlr_neg(m, train_dataset, valid_dataset, args, data_collator):
    loss = MultipleNegativesRankingLoss(m)

    if valid_dataset is not None:
        evaluator = get_ret_eval(valid_dataset)

    trainer = SentenceTransformerTrainer(
        model = m,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=loss,
        evaluator=evaluator,
        args=args,
        data_collator=data_collator
    )

    return trainer

def trainer_cl_3classes(m, train_dataset, valid_dataset, args):
    loss = ContrastiveLoss(m)

    evaluator = MyRecallEval(valid_dataset)

    trainer = SentenceTransformerTrainer(
        model = m,
        train_dataset=train_dataset,
        eval_dataset=filter_neg(valid_dataset),
        loss=loss,
        evaluator=evaluator,
        args=args
    )

    return trainer

def trainer_mnr(m, train_dataset, valid_dataset, args):
    td = {''}
    loss = MultipleNegativesRankingLoss(m)

    evaluator = get_ret_eval(valid_dataset)

    trainer = SentenceTransformerTrainer(
        model = m,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=loss,
        evaluator=evaluator,
        args=args
    )

    return trainer

def trainer3(m, train_dataset, valid_dataset, args):
    loss = TripletLoss(m)

    evaluator = MyRecallEval(ds_abs(valid_dataset))

    trainer = SentenceTransformerTrainer(
        model = m,
        train_dataset=triple(train_dataset),
        eval_dataset=triple(valid_dataset),
        loss=loss,
        evaluator=evaluator,
        args=args
    )

    return trainer
