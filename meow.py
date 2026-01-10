from datasets import Dataset, load_dataset
import pandas as pd

train_data_path = "./data/English dataset/train.jsonl"
test_data_path = "./data/English dataset/test.jsonl"

train_data = pd.DataFrame(load_dataset("json", data_files=train_data_path)["train"])
test_dataset = pd.DataFrame(load_dataset("json", data_files=test_data_path)["train"])

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

dss = ds.train_test_split(valid_split)
train_dataset = dss['train']
valid_dataset = dss['test']
test_dataset = Dataset.from_pandas(test_dataset)
test_dataset = test_dataset.select_columns(["hypothesis", "premise", "label"])

