import datetime
import html
import os
import csv
import re
from dotenv import load_dotenv
from datetime import datetime
from elasticsearch import Elasticsearch
from ai_elastic import HybridRetriever

def _split_path(dotted_key):
    """Split a dotted key on '.' while treating '[...]' as atomic, so a filter
    expression like 'Type.Id=<uuid>' inside brackets isn't split on its own dot."""
    parts = []
    current = ""
    depth = 0
    for ch in dotted_key:
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
        elif ch == "." and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return parts

def _get_nested(d, dotted_key):
    """Traverse a nested dict/list using a dot-separated key.
    Supports array indexing: 'IdentifierDoi[0]' or 'IdentifierScopusId[0] or 'Authors[0].Name'.
    Supports wildcard: 'Persons[*].PersonData.DisplayName' collects all values joined by ' ; '.
    'Persons[*].PersonData.IdentifierCid[0]' collects all values joined by ' ; '.
    Supports filtering: 'Categories[Type.Id=<uuid>].NameEng' collects values only from array
    items whose nested field matches the given value, joined by ' ; '.
    """
    parts = _split_path(dotted_key)
    for i, part in enumerate(parts):
        m_all = re.fullmatch(r"(\w+)\[\*\]", part)
        m_filter = re.fullmatch(r"(\w+)\[(\w+(?:\.\w+)*)=([^\]=]+)\]", part)
        m_idx = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        if m_all:
            if not isinstance(d, dict):
                return ""
            lst = d.get(m_all.group(1), [])
            if not isinstance(lst, list):
                return ""
            remaining = ".".join(parts[i + 1:])
            values = [_get_nested(item, remaining) for item in lst] if remaining else lst
            return " ; ".join(str(v) for v in values if v)
        elif m_filter:
            key, filter_key, filter_val = m_filter.group(1), m_filter.group(2), m_filter.group(3)
            if not isinstance(d, dict):
                return ""
            lst = d.get(key, [])
            if not isinstance(lst, list):
                return ""
            matched = [item for item in lst if str(_get_nested(item, filter_key)) == filter_val]
            remaining = ".".join(parts[i + 1:])
            values = [_get_nested(item, remaining) for item in matched] if remaining else matched
            return " ; ".join(str(v) for v in values if v)
        elif m_idx:
            key, idx = m_idx.group(1), int(m_idx.group(2))
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
    timeout=120,
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
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", 5000))
STATIC_QUERY = os.environ.get("STATIC_QUERY") or None
STATIC_WEIGHT = float(os.environ.get("STATIC_WEIGHT", 1.0))

# --- Quick keyword-only smoke test, DEBUG ---
if SEARCH_MODE != "semantic":
    print(f"[debug] running keyword search for: {QUERY!r}")
    try:
        resp = es.search(
            index=os.environ["ES_INDEX"],
            body={
                "query": {
                    "bool": {
                        "must": {"simple_query_string": {"query": QUERY, "fields": ["Title^3", "Abstract^2", "Categories.NameEng^2", "Keywords^2"]}},
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

# Retrieve results with both methods (plus an optional static/predefined pool), combine them with RRF and write to file
if STATIC_QUERY:
    print(f"[debug] including static pool for: {STATIC_QUERY!r} (weight={STATIC_WEIGHT})")
results = retriever.search(QUERY, top_k=MAX_RESULTS, mode=SEARCH_MODE, static_query=STATIC_QUERY, weights=(1.0, 1.0, STATIC_WEIGHT))

CSV_FIELDS = ["Id", "Title", "IdentifierDoi[0]", "IdentifierScopusId[0]", "Persons[*].PersonData.DisplayName", "Persons[*].PersonData.IdentifierCid[0]", "Abstract", "Year", "PublicationType.NameEng",
              "Categories[Type.Id=fba59577-7c91-4a65-9154-7fd8b630f81a].NameEng"]
CSV_HEADER = ["Id", "Title", "DOI", "Scopus ID", "Authors", "CID", "Abstract", "Year", "PublicationType", "Chalmers AoA"]
OUTFILE_CSV = os.environ.get('OUTFILE_CSV', "results") + f".{datetime.now().strftime('%Y%m%d.%H%M%S')}.csv"

print(f"\nRESULTS:\n")

with open(OUTFILE_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, CSV_FIELDS + ["rrf_score", "matched_methods"])
    writer.writer.writerow(CSV_HEADER + ["RRF Score", "Matched Methods"])

    def _to_es_field(f):
        # A filter bracket (e.g. 'Categories[Type.Id=...].NameEng') needs sibling
        # fields (Type.Id) beyond the trailing subpath to survive the fetch, so
        # request the whole base array instead of narrowing _source to NameEng.
        m = re.search(r"\[[^\]]*=[^\]]*\]", f)
        return f[:m.start()] if m else re.sub(r"\[[^\]]*\]", "", f)

    es_fields = list(dict.fromkeys(_to_es_field(f) for f in CSV_FIELDS))
    all_doc_ids = [r["doc_id"] for r in results]
    print(f"Fetching {len(all_doc_ids)} records from ES...")
    records_by_id = retriever.fetch_records(all_doc_ids, fields=es_fields)

    for r in results:
        methods = ["keyword" if 0 in r["matched_methods"] else None,
                   "semantic" if 1 in r["matched_methods"] else None,
                   "static" if 2 in r["matched_methods"] else None]
        methods = [m for m in methods if m]
        print(f"{r['doc_id']:20s} score={r['rrf_score']:.4f} via {'+'.join(methods)}")

        record = records_by_id.get(r["doc_id"]) or {}
        # Only include publications from the specified publication year onward (e.g. 2018-)
        if (record.get('Year') or 0) < int(os.environ.get("START_YEAR", 2014)):
            print(f"  Skipping record!")
            continue
        row = {f: _get_nested(record, f) for f in CSV_FIELDS}
        # Clean title text for CSV output (remove HTML tags, unescape entities, normalize whitespace)
        if row.get("Title"):
            row["Title"] = re.sub(r"<[^>]+>", " ", row["Title"])
            row["Title"] = html.unescape(row["Title"])
            row["Title"] = re.sub(r"\s+", " ", row["Title"]).strip()
        # Clean abstract text for CSV output (remove HTML tags, unescape entities, normalize whitespace)
        if row.get("Abstract"):
            abstract = re.sub(r"<[^>]+>", " ", row["Abstract"])
            abstract = html.unescape(abstract)
            row["Abstract"] = re.sub(r"\s+", " ", abstract).strip()
        # Scopus stores bare IDs, prefix with EID namespace so it matches Scopus's own EID format
        if row.get("IdentifierScopusId[0]"):
            row["IdentifierScopusId[0]"] = "2-s2.0-" + row["IdentifierScopusId[0]"]
            
        # Write the row to CSV with additional RRF score and matched methods
        writer.writerow({
            **row,
            "rrf_score": round(r["rrf_score"], 6),
            "matched_methods": "+".join(methods),
        })

print(f"\nDone! Results written to {OUTFILE_CSV}\n")
