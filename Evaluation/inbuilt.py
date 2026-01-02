from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict

def get_ret_eval(test_dataset):
    corpus = dict(zip(test_dataset['premise'], test_dataset['premise']))
    queries = dict(zip(test_dataset['hypothesis'], test_dataset['hypothesis']))
    relevant_docs = defaultdict(list)

    for k in range(len(test_dataset)):
        if test_dataset['label'][k] > 0:
            relevant_docs[test_dataset['hypothesis'][k]].append(test_dataset['premise'][k])
    

    inf_ret_ev = InformationRetrievalEvaluator(
        queries= queries,
        corpus = corpus,
        relevant_docs = relevant_docs,
        #similarity_fn_names= ["cosine"],
        show_progress_bar=True,
        batch_size= 16,
        #main_score_function="Recall@10"
    )

    return inf_ret_ev

from sentence_transformers.evaluation import BinaryClassificationEvaluator

def get_bin_eval(test_dataset):
    """
        BinnaryClassification returns: F1, Percision, Recall, Avg Percision, Matthews Correlation, 
    """
    bin_acc_ev = BinaryClassificationEvaluator(
        sentences1= test_dataset['hypothesis'],
        sentences2= test_dataset['premise'],
        labels= test_dataset['label'],
        similarity_fn_names= ["cosine", "dot"],
        show_progress_bar= True,
        batch_size= 16
    )
    return bin_acc_ev