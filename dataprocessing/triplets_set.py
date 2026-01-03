import pandas as pd
import numpy as np
import datasets
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from collections import defaultdict

from datasets import Dataset
import itertools
def triple(data):
    relevant_docs = defaultdict(list)
    irelevant_docs = defaultdict(list)

    for k in range(len(data)):
        if data['label'][k] > 0:
            relevant_docs[data['hypothesis'][k]].append(data['premise'][k])
        elif data['label'][k] < 0:
            irelevant_docs[data["hypothesis"][k]].append(data['premise'][k])

    triplets = {'anchor': [], 'positive': [], 'negative': []}
    for i, h in enumerate(set(data['hypothesis'])):
        for t, f in itertools.product(relevant_docs[h], irelevant_docs[h]):
            triplets["anchor"].append(h)
            triplets["positive"].append(t)
            triplets["negative"].append(f)

    meow = Dataset.from_dict(triplets)
    print("meow",len(meow))
    return meow
def ds_abs(data):
    d = {'hypothesis':data['hypothesis'], 'premise':data['premise'], 'label':[]}
    
    for k in range(len(data)):
        d['label'].append(max(0,data['label'][k]))

    meow = Dataset.from_dict(d)
    print("meow",len(meow))
    return meow
def get_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    train_data_path = "./data/English dataset/train.jsonl"
    test_data_path = "./data/English dataset/test.jsonl"

    train_data = pd.DataFrame(datasets.load_dataset("json", data_files=train_data_path)["train"])
    test_dataset = pd.DataFrame(datasets.load_dataset("json", data_files=test_data_path)["train"])

    label_map = {"Contradiction": 1, "Entailment": -1, "NotMentioned": 0}
    label_map2 = {"Contradiction": 1, "Entailment": 0, "NotMentioned": 0}

    train_data["label"] = train_data["label"].map(label_map)
    test_dataset["label"] = test_dataset["label"].map(label_map2)

    train_data = train_data.drop("doc_id", axis=1)
    train_data = train_data.drop("key", axis=1)
    test_dataset = test_dataset.drop("doc_id", axis=1)
    test_dataset = test_dataset.drop("key", axis=1)

    train_data["label"].value_counts(normalize=True)


    ds = Dataset.from_pandas(train_data)
    ds = ds.select_columns(["hypothesis", "premise", "label"])
    ds = ds.select_columns(["hypothesis", "premise", "label"])

    dss = ds.train_test_split(0.1, seed=42)
    train_dataset = dss['train']
    train_dataset = train_dataset
    valid_dataset = dss['test']
    test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.select_columns(["hypothesis", "premise", "label"])

    return train_dataset, valid_dataset, test_dataset