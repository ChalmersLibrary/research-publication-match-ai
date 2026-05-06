import os

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from dotenv import load_dotenv
from collections import defaultdict

# Hybrid Retrieval: Combining ES Keyword Scores with FAISS Semantic Scores

class HybridRetriever:
    def __init__(self, es_client, es_index, faiss_index_path, 
                 metadata_path, embedding_model="allenai/specter2_base"):
        self.es = es_client
        self.es_index = es_index
        
        # Load FAISS index and metadata
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.metadata = []
        with open(metadata_path) as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        # Load embedding model (must match what built the index)
        self.model = SentenceTransformer(embedding_model)
    
    def keyword_search(self, query_text, top_k=100):
        """Returns list of (doc_id, rank) tuples from ES."""
        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["Title^3", "Abstract^2", "Categories.NameEng^2", "Keywords^2"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "range": {"Year": {"gte": os.environ.get("START_YEAR", 2014)}}
                    }
                }
            },
            "size": top_k,
        }
        response = self.es.search(index=self.es_index, body=body)
        return [
            (hit["_source"].get("Id") or hit["_id"], rank, hit["_score"], hit["_source"].get("Title", ""))
            for rank, hit in enumerate(response["hits"]["hits"], start=1)
        ]
    
    def semantic_search(self, query_text, top_k=100):
        """Returns list of (doc_id, rank, score) tuples from FAISS."""
        query_vec = self.model.encode([query_text]).astype("float32")
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.faiss_index.search(query_vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            doc_id = self.metadata[idx]["id"]
            results.append((doc_id, rank, float(score)))
        return results
    
    def reciprocal_rank_fusion(self, result_lists, k=60, weights=None):
        """
        Merge multiple ranked result lists using RRF.
        
        result_lists: list of lists, each containing (doc_id, rank, score) tuples
        k: smoothing constant (60 is standard)
        weights: optional per-list weights (default: equal)
        """
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        rrf_scores = defaultdict(float)
        seen_in = defaultdict(list)
        titles = {}

        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx]
            for doc_id, rank, _, *rest in results:
                rrf_scores[doc_id] += weight * (1.0 / (k + rank))
                seen_in[doc_id].append(list_idx)
                if rest and rest[0]:
                    titles[doc_id] = rest[0]

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return [
            {"doc_id": doc_id, "rrf_score": score,
             "matched_methods": seen_in[doc_id],
             "title": titles.get(doc_id, "")}
            for doc_id, score in ranked
        ]
    
    def search(self, query_text, top_k=50, candidates_per_method=100, 
               weights=(1.0, 1.0)):
        """
        Hybrid search: retrieve from both, fuse, return top_k.
        
        weights: (keyword_weight, semantic_weight)
        """
        kw_results = self.keyword_search(query_text, top_k=candidates_per_method)
        sem_results = self.semantic_search(query_text, top_k=candidates_per_method)
        
        fused = self.reciprocal_rank_fusion(
            [kw_results, sem_results],
            weights=list(weights)
        )
        return fused[:top_k]

    def fetch_record(self, doc_id, fields=None):
        if fields is None:
            env_val = os.environ.get("FETCH_FIELDS")
            fields = env_val.split(",") if env_val else ("Id", "Title", "Year")
        """Fetch selected fields for a single doc_id. Returns a dict or None."""
        body = {
            "query": {"term": {"Id": doc_id}},
            "_source": list(fields),
            "size": 1,
        }
        print(f"[fetch_record] doc_id={doc_id!r}  query={body['query']}")
        response = self.es.search(index=self.es_index, body=body)
        hits = response["hits"]["hits"]
        print(f"[fetch_record] total hits={response['hits']['total']}  returned={len(hits)}")
        if hits:
            print(f"[fetch_record] _id={hits[0]['_id']}  _source keys={list(hits[0]['_source'].keys())}")
        return hits[0]["_source"] if hits else None
