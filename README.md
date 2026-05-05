**research-publication-match-ai**

App for locating publications within specific subject(s) and/or matching other criteria in [Chalmers Research CRIS](https://research.chalmers.se), by querying both ES for keyword hits and vector store for semantic hits — and then merge the results with reciprocal rank fusion.     
Uses [FAISS](https://github.com/facebookresearch/faiss) and the [allenai/specter2](https://huggingface.co/allenai/specter2) model for embedding.    

Requirements   

* Read access to an elastic research-publications index.
* Python 3.x with required modules
    - FAISS, Sentence-transformers, Elasticsearch (<7.16.3), Peft

Setup   

- Install python libraries    
``pip install faiss-cpu sentence-transformers python-dotenv peft``    

- Install working module for ES6.    
``pip install elasticsearch-7.16.3``    

- Create local vector store (FAISS and jsonl indexes) for semantic search (may take a while, must be re-run if the ES index content change)       
``python build_index.py``.   

- ``python main.py``   

Output    

The current version only return the top 50 hits, with publication ID, ranking score and method (keyword and/or semantic). This can be changed inside the script.   

Known issues     

* Different versions of python elasticsearch module can cause errors like: TypeError: index() got an unexpected keyword argument 'document'.  
Using elasticsearch-7.16.3 should work (required for ES 6).
* Encoding errors (Linux only?) can be handled by running this script as: 
PYTHONIOENCODING=utf-8 python main.py
