import random
from datasets import Dataset
import pandas as pd
import datasets
from datasets import Dataset
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from torch._tensor import Tensor
from typing import Any

class MyDataCollator(SentenceTransformerDataCollator):
    universal_negatives = ['', '']
    negatives_per_batch = 4

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        # Add 2 negative texts to each feature as a list
        k = list(features[0].keys())[1]
        for i, feature in enumerate(features):
            features[i][k] = [features[i][k]] + random.sample(
            self.universal_negatives, 
            min(self.negatives_per_batch, len(self.universal_negatives))
        )
        #print(features[i][k])
        batch = super().__call__(features)
        
        return batch

def filter_neg(data):
    h = len(set(data['hypothesis']))
    p = len(set(data['premise']))
    hmm = dict(zip(set(data['premise']), range(p)))
    d = {'hypothesis':[], 'premise':[]}
    print("hehehehe") # This is what the fox say
    for k in range(len(data)):
        if data['label'][k] == 1:
            d['hypothesis'].append(data['hypothesis'][k])
            d['premise'].append(data['premise'][k])
        else:
            hmm[data['premise'][k]] += 1
    uni = []
    for k, v in hmm.items():
        if v >= h:
            uni.append(k)
    print("conving")
    #meow = np.array((d['hypothesis'], d['premise']))
    return (d, uni)


def get_data(model, valid=False):
    train_data_path = "./data/English dataset/train.jsonl"
    test_data_path = "./data/English dataset/test.jsonl"

    train_data = pd.DataFrame(datasets.load_dataset("json", data_files=train_data_path)["train"])
    test_dataset = pd.DataFrame(datasets.load_dataset("json", data_files=test_data_path)["train"])

    label_map = {"Contradiction": 1, "Entailment": 0, "NotMentioned": 0}
    train_data["label"] = train_data["label"].map(label_map)
    test_dataset["label"] = test_dataset["label"].map(label_map)

    train_data = train_data.drop("doc_id", axis=1)
    train_data = train_data.drop("key", axis=1)
    test_dataset = test_dataset.drop("doc_id", axis=1)
    test_dataset = test_dataset.drop("key", axis=1)

    train_data["label"].value_counts(normalize=True)

    ds = Dataset.from_pandas(train_data)
    ds = ds.select_columns(["hypothesis", "premise", "label"])

    if valid:        
        dss = ds.train_test_split(0.2, seed=42)
        train_dataset = dss['train']
        dsss = dss['test']
        dsss = dsss.train_test_split(0.5, seed=42)
        valid_dataset = dsss['train']
        test_dataset = dsss['test']
        test_dataset = test_dataset.select_columns(["hypothesis", "premise", "label"])

        return train_dataset, valid_dataset, test_dataset

    dss = ds.train_test_split(0.1, seed=42)
    train_dataset = dss['train']
    train_dataset, universal_negatives = filter_neg(train_dataset)
    train_dataset = Dataset.from_dict(train_dataset)
    data_collator = MyDataCollator(model.tokenize)
    data_collator.universal_negatives = universal_negatives
    valid_dataset = dss['test']
    test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.select_columns(["hypothesis", "premise", "label"])
    #train_dataset = train_dataset.map(hihi)
    #valid_dataset = valid_dataset.map(hihi)
    #test_dataset = test_dataset.map(hihi)
    return train_dataset, valid_dataset, test_dataset, data_collator