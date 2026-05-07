import datetime
import os
import csv
import re
from dotenv import load_dotenv
from datetime import datetime
from elasticsearch import Elasticsearch
from ai_elastic import HybridRetriever

def _get_nested(d, dotted_key):
    """Traverse a nested dict/list using a dot-separated key.
    Supports array indexing: 'IdentifierDoi[0]' or 'Authors[0].Name'.
    """
    for part in dotted_key.split("."):
        m = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if not isinstance(d, dict):
                return ""
            lst = d.get(key, [])
            d = lst[idx] if isinstance(lst, list) and idx < len(lst) else ""
        else:
            if not isinstance(d, dict):
                return ""
            d = d.get(part, "")
    return d

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

#QUERY = "maritime marine shipping seafood aquaculture blue bioeconomy ocean currents wrecks ships boats"
QUERY = os.environ.get("QUERY")
SEARCH_MODE = os.environ.get("SEARCH_MODE", "hybrid")

# --- Quick keyword-only smoke test, DEBUG ---
if SEARCH_MODE != "semantic":
    print(f"[debug] running keyword search for: {QUERY!r}")
    try:
        resp = es.search(
            index=os.environ["ES_INDEX"],
            body={
                "query": {
                    "bool": {
                        "must": {"multi_match": {"query": QUERY, "fields": ["Title^3", "Abstract^2", "Categories.NameEng^2", "Keywords^2"]}},
                        "filter": [
                            {"range": {"Year": {"gte": os.environ.get("START_YEAR", 2014)}}},
                            {"term": {"NeedsAttention": False}},
                            {"term": {"IsDraft": False}},
                            {"term": {"IsDeleted": False}},
                        ],
                    }
                },
                "size": 5,
            },
        )
        hits = resp["hits"]["hits"]
        total = resp["hits"]["total"]
        total_count = total["value"] if isinstance(total, dict) else total
        print(f"\n[debug] keyword hits: {total_count} in total.\n")
    except Exception as e:
        print(f"[debug] keyword search failed: {e}")
        raise

# Full hybrid search (requires publications.faiss + metadata.jsonl) ---
retriever = HybridRetriever(
    es_client=es,
    es_index=os.environ["ES_INDEX"],
    faiss_index_path="publications.faiss",
    metadata_path="metadata.jsonl"
)

# Retrieve results with both methods, combine them with RRF and write to file

results = retriever.search(QUERY, top_k=5000, mode=SEARCH_MODE)

CSV_FIELDS = ["Id", "Title", "IdentifierDoi[0]", "Year", "PublicationType.NameEng"]
OUTFILE_CSV = os.environ.get('OUTFILE_CSV', "results") + f".{datetime.now().strftime('%Y%m%d.%H%M%S')}.csv"

print(f"\nRESULTS:\n")

with open(OUTFILE_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, CSV_FIELDS + ["rrf_score", "matched_methods"])
    writer.writeheader()

    for r in results:
        methods = ["keyword" if 0 in r["matched_methods"] else None,
                   "semantic" if 1 in r["matched_methods"] else None]
        methods = [m for m in methods if m]
        print(f"{r['doc_id']:20s} score={r['rrf_score']:.4f} via {'+'.join(methods)}")

        es_fields = list(dict.fromkeys(re.sub(r"\[\d+\]", "", f) for f in CSV_FIELDS))
        record = retriever.fetch_record(r["doc_id"], fields=es_fields) or {}
        # Only include publications from the specified publication year onward (e.g. 2018-)
        if (record.get('Year') or 0) < int(os.environ.get("START_YEAR", 2014)):
            print(f"  Skipping record!")
            continue
        writer.writerow({
            **{f: _get_nested(record, f) for f in CSV_FIELDS},
            "rrf_score": round(r["rrf_score"], 6),
            "matched_methods": "+".join(methods),
        })

print(f"\nDone! Results written to {OUTFILE_CSV}\n")
    

