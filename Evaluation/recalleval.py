# Custom evaluator
from sentence_transformers.evaluation import SentenceEvaluator
from collections import defaultdict
from sentence_transformers.util import cos_sim
import torch
import numpy as np
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

class MyRecallEval(SentenceEvaluator):
    def structure(test_dataset):
        corpus = dict(zip(test_dataset['premise'], test_dataset['premise']))
        queries = dict(zip(test_dataset['hypothesis'], test_dataset['hypothesis']))
        relevant_docs = defaultdict(list)

        for k in range(len(test_dataset)):
            if test_dataset['label'][k] > 0:
                relevant_docs[test_dataset['hypothesis'][k]].append(test_dataset['premise'][k])

        return (queries, corpus, relevant_docs)
        
    def __init__(self, data, recall_ks=[10], cluster_k=10,cluster_min_hits=1,name: str = "", main_similarity=SimilarityFunction.COSINE):
        super().__init__()
        hypotheses, premises, relevant_premises = MyRecallEval.structure(data)
        self.hypotheses = hypotheses
        self.premises = premises
        self.relevant_premises = relevant_premises
        self.recall_ks = recall_ks
        self.cluster_k = cluster_k
        self.cluster_min_hits = cluster_min_hits
        self.name = name
        self.main_similarity = SimilarityFunction.to_similarity_fn(main_similarity)
        self.greater_is_better = True
        self.primary_metric = f"recall@{max(recall_ks)}"

        # Fixed ordering (important!)
        self.hyp_ids = list(hypotheses.keys())
        self.premise_ids = list(premises.keys())

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # 1. Encode
        hyp_texts = [self.hypotheses[h] for h in self.hyp_ids]
        prem_texts = [self.premises[p] for p in self.premise_ids]

        hyp_emb = model.encode(hyp_texts, convert_to_tensor=True, normalize_embeddings=True)
        prem_emb = model.encode(prem_texts, convert_to_tensor=True, normalize_embeddings=True)

        # 2. Similarity matrix
        # TODO should we use cos_sim?
        scores = self.main_similarity(hyp_emb, prem_emb)  # shape: [num_hyp, num_prem]

        recalls = {k: [] for k in self.recall_ks}
        normrecalls = {k: [] for k in self.recall_ks}

        cluster_success = []

        # 3. Per-hypothesis evaluation
        for i, hyp_id in enumerate(self.hyp_ids):
            relevant = self.relevant_premises[hyp_id]
            if not relevant:
                continue

            relevant_idx = {self.premise_ids.index(pid) for pid in relevant}

            ranked = torch.argsort(scores[i], descending=True)

            for k in self.recall_ks:
                topk = ranked[:k].tolist()
                hits = len(set(topk) & relevant_idx)
                normrecalls[k].append(hits / min(k,len(relevant_idx)))
                recalls[k].append(hits / len(relevant_idx))

            # Cluster recall
            top_cluster = ranked[: self.cluster_k].tolist()
            hits = len(set(top_cluster) & relevant_idx)
            cluster_success.append(hits >= self.cluster_min_hits)
            print(cluster_success)
        # 4. Aggregate metrics
        metrics = {
            f"recall@{k}": float(np.mean(recalls[k])) for k in self.recall_ks
        }
        for k in self.recall_ks:
            metrics[f"normalized_recall@{k}"] = float(np.mean(normrecalls[k]))
        metrics["cluster_recall"] = float(np.mean(cluster_success))

        # Optional: store in model card
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        # Instead of:
        # return self.prefix_name_to_metrics(metrics, self.name)

        # Use:
        if self.name:
            for key in list(metrics.keys()):
                metrics[f"{self.name}_{key}"] = metrics.pop(key)
        return metrics
        #return self.prefix_name_to_metrics(metrics, self.name)