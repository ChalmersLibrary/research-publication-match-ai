import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from ai_elastic import HybridRetriever

# Use as python main.py to run the search demo. Make sure to set up your .env file with the appropriate ES connection details and index name.
# Regs:
# pip install faiss-cpu sentence-transformers elasticsearch python-dotenv peft
# ES version...7.x is recommended for compatibility with the code. Adjust as needed for newer versions.
#
# Create FAISS index and jsonl file with python build_index.py:
# It will:
# Scroll all documents from ES (using the index in your .env)
# Embed them with allenai-specter in the SPECTER format (title [SEP] abstract)
# Save publications.faiss and metadata.jsonl in the current directory
# After it finishes, python main.py should work end-to-end. The index only needs to be rebuilt if the ES data changes.
#

load_dotenv()

es = Elasticsearch(
    [os.environ["ES_URL"]],
    http_auth=(os.environ["ES_UID"], os.environ["ES_PW"]) if os.environ.get("ES_UID") else None,
)

# --- ES connection check ---
print(f"[debug] connecting to {os.environ['ES_URL']}, index={os.environ['ES_INDEX']}")
try:
    info = es.info()
    print(f"[debug] ES cluster: {info['cluster_name']} (v{info['version']['number']})")
except Exception as e:
    print(f"[debug] ES connection failed: {e}")
    raise

QUERY = "maritime marine shipping seafood aquaculture blue bioeconomy ocean currents wrecks ships boats"

# --- Quick keyword-only smoke test, DEBUG ---
# print(f"[debug] running keyword search for: {QUERY!r}")
# try:
#     resp = es.search(
#         index=os.environ["ES_INDEX"],
#         body={"query": {"multi_match": {"query": QUERY, "fields": ["Title^3", "Abstract^2"]}}, "size": 5},
#     )
#     hits = resp["hits"]["hits"]
#     total = resp["hits"]["total"]
#     total_count = total["value"] if isinstance(total, dict) else total
#     print(f"[debug] keyword hits: {total_count} total, showing top {len(hits)}")
#     for h in hits:
#         print(f"  score={h['_score']:.4f}  id={h['_source'].get('id')}  title={h['_source'].get('title') or h['_source'].get('Title')}")
# except Exception as e:
#     print(f"[debug] keyword search failed: {e}")
#     raise

# --- Full hybrid search (requires publications.faiss + metadata.jsonl) ---
retriever = HybridRetriever(
    es_client=es,
    es_index=os.environ["ES_INDEX"],
    faiss_index_path="publications.faiss",
    metadata_path="metadata.jsonl"
)

results = retriever.search(QUERY, top_k=50)

print("RESULTS:")

for r in results:
    methods = ["keyword" if 0 in r["matched_methods"] else None,
               "semantic" if 1 in r["matched_methods"] else None]
    methods = [m for m in methods if m]
    print(f"{r['doc_id']:20s} score={r['rrf_score']:.4f} via {'+'.join(methods)}")
    #print(f"{r['doc_id']:20s} score={r['rrf_score']:.4f} via {'+'.join(methods)}  {r['title']}")
