import pandas as pd
import numpy as np
import datasets
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from datasets import Dataset

def pair(data):
    d = {'hypothesis':[], 'premise':[], 'label':[]}
    
    for k in range(len(data)):
        if data['label'][k] != 0:
            d['hypothesis'].append(data['hypothesis'][k])
            d['premise'].append(data['premise'][k])
            d['label'].append(max(0,data['label'][k]))

    meow = Dataset.from_dict(d)
    print("meow",len(meow))
    return meow
def get_data(valid=False):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    lemmatizer = WordNetLemmatizer()
    train_data_path = "./data/English dataset/train.jsonl"
    test_data_path = "./data/English dataset/test.jsonl"

    def preprocess_text(text): # From the labs
        # Tokenize the text into words
        words = word_tokenize(text.lower())  # Convert text to lowercase

        # Remove punctuation
        table = str.maketrans('', '', string.punctuation)
        words = [word.translate(table) for word in words if word.isalpha()]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Lemmatization
        #lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Join the words back into a string
        preprocessed_text = ' '.join(words)
        return preprocessed_text
        
    def hihi(rows):
        p = preprocess_text(rows['premise'])
        h = preprocess_text(rows['hypothesis'])
        return {'hypothesis': h, 'premise': p}

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

    from datasets import Dataset

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
    valid_dataset = dss['test']
    test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.select_columns(["hypothesis", "premise", "label"])
    train_dataset = train_dataset.map(hihi)
    valid_dataset = valid_dataset.map(hihi)
    test_dataset = test_dataset.map(hihi)
    return train_dataset, valid_dataset, test_dataset

def get_data_seperate_irrelevant():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    train_data_path = "./data/English dataset/train.jsonl"
    test_data_path = "./data/English dataset/test.jsonl"

    def preprocess_text(text): # From the labs
        # Tokenize the text into words
        words = word_tokenize(text.lower())  # Convert text to lowercase

        # Remove punctuation
        table = str.maketrans('', '', string.punctuation)
        words = [word.translate(table) for word in words if word.isalpha()]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # Lemmatization
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Join the words back into a string
        preprocessed_text = ' '.join(lemmatized_words)
        return preprocessed_text

    train_data = pd.DataFrame(datasets.load_dataset("json", data_files=train_data_path)["train"])
    test_dataset = pd.DataFrame(datasets.load_dataset("json", data_files=test_data_path)["train"])

    label_map = {"Contradiction": 1, "Entailment": 0, "NotMentioned": -1}
    train_data["label"] = train_data["label"].map(label_map)
    test_dataset["label"] = test_dataset["label"].map(label_map)

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
    valid_dataset = dss['test']
    test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.select_columns(["hypothesis", "premise", "label"])

    return pair(train_dataset), pair(valid_dataset), test_dataset