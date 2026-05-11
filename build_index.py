"""
Build publications.faiss and metadata.jsonl from the Elasticsearch index.
Run once before using main.py:  python build_index.py
"""
import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()

ES_URL   = os.environ["ES_URL"]
ES_UID   = os.environ.get("ES_UID")
ES_PW    = os.environ.get("ES_PW")
ES_INDEX = os.environ["ES_INDEX"]
EMBEDDING_MODEL   = "allenai/specter2_base"
FAISS_INDEX_PATH  = "publications.faiss"
METADATA_PATH     = "metadata.jsonl"
BATCH_SIZE        = 256
SCROLL_SIZE       = 500
SCROLL_TTL        = "5m"

es = Elasticsearch(
    [ES_URL],
    http_auth=(ES_UID, ES_PW) if ES_UID else None,
)

print(f"[build] connected to {ES_URL}, index={ES_INDEX}")
print(f"[build] loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# Scroll all documents from ES
print("[build] scrolling documents from ES...")
resp = es.search(
    index=ES_INDEX,
    scroll=SCROLL_TTL,
    size=SCROLL_SIZE,
    body={
        "query": {
            "bool": {
                "filter": [
                    {"term": {"NeedsAttention": False}},
                    {"term": {"IsDraft": False}},
                    {"term": {"IsDeleted": False}},
                ]
            }
        },
        "_source": ["id", "Title", "Abstract", "Keywords"],
    },
)
scroll_id = resp["_scroll_id"]

docs = []
while True:
    hits = resp["hits"]["hits"]
    if not hits:
        break
    for h in hits:
        src = h["_source"]
        docs.append({
            "id": src.get("id", h["_id"]),
            "title": src.get("Title", ""),
            "abstract": src.get("Abstract", ""),
            "keywords": src.get("Keywords.Value", []),
        })
    print(f"[build]   fetched {len(docs)} docs so far ...", end="\r")
    resp = es.scroll(scroll_id=scroll_id, scroll=SCROLL_TTL)
    scroll_id = resp["_scroll_id"]

es.clear_scroll(scroll_id=scroll_id)
print(f"\n[build] total documents: {len(docs)}")

# Build SPECTER-style input strings: "title [SEP] abstract"
texts = [f"{d['title']} [SEP] {d['abstract']}" for d in docs]

# Embed in batches
print("[build] embedding documents ...")
all_vecs = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    vecs = model.encode(batch, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vecs)
    all_vecs.append(vecs)
    print(f"[build]   embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")

print()
embeddings = np.vstack(all_vecs)

# Build FAISS flat inner-product index (equivalent to cosine after L2 norm)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"[build] saved FAISS index ({index.ntotal} vectors, dim={dim}) -> {FAISS_INDEX_PATH}")

# Save metadata
with open(METADATA_PATH, "w") as f:
    for d in docs:
        f.write(json.dumps({"id": d["id"]}) + "\n")
print(f"[build] saved metadata -> {METADATA_PATH}")
