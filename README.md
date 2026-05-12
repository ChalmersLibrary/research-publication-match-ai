# research-publication-match-ai

App for locating publications within specific subject(s) and/or matching other criteria in [Chalmers Research CRIS](https://research.chalmers.se), by querying both ES for keyword hits and vector store for semantic hits — and then merge the results with combined rankings (reciprocal rank fusion).     
Uses [FAISS](https://github.com/facebookresearch/faiss) and the [allenai/specter2_base](https://huggingface.co/allenai/specter2) model for embeddings.       
    
It is also possible to run a semantic or keyword search only. See SEARCH_MODE in *Setup and run* below.    

Queries are issued against the title, abstract, keyword and category (subject) fields. This (and other things) can be changed inside the script.    

### Requirements  

* Read access to an elasticsearch Chalmers research-publications index (use a static index if possible, preferred both for indexing and performance)   
* Python 3.x with required packages:
    - FAISS, SentenceTransformer, Elasticsearch (<=7.16.3), Peft
* A decent PC (Windows, Mac, Linux) with at least 8 GB RAM and a quadcore CPU (primarily for generating the vector store).            

### Setup and run   

- Install python packages    
``pip install faiss-cpu sentence_transformers python-dotenv peft os csv re``    

- Install a working package for Elasticsearch 6.x (for compability)    
``pip install elasticsearch-7.16.3``

- Create an **.env** file with local settings, in the current directory, using *env_example* as template (note the format of the examples in this file):    
    - ES_URL - elasticsearch base URL   
    - ES_UID - elasticsearch user ID   
    - ES_PW - elasticsearch password   
    - ES_INDEX - name of the elastic index (eg. *research-publications-static-20260101*)   
    - OUTFILE_CSV - name of output CSV file (default: *results*)    
    - QUERY - query string (keywords, space separated) to be used, eg. *maritime marine shipping seafood aquaculture blue bioeconomy ocean currents*
    - FETCH_FIELDS - fields that should be retrieved from Chalmers CRIS and included in the output, eg. *Id,Title,IdentifierDoi[0],Year,Abstract,PublicationType.NameEng*   
    - START_YEAR - only include publications from this year forwards (default: *2014*)
    - POOL_SIZE - how many publication records should be handled at a time in each pool when searching (keyword, semantic). Setting this too high could cause timeout errors. (default: *1000*)   
    - SEARCH_MODE - *hybrid* (both keyword and semantic search, with RRF), *semantic* (only) or *keyword* (only). (default: *hybrid*)       

- Create local vector store (FAISS and jsonl indexes) for semantic search (may take a while and must be re-run if the ES index content changes, but not if just modifying the query or search filters)       
``python build_index.py``   

- Run the script    
``python main.py``   

### Output  

The output is written to a local CSV file in the current directory (see *main.py* for details and adjust if needed). File name is specified in the *.env* file (default: *results[.YYYYMMDD.hhmmss.csv]*).   

The current (proof of concept) version return Publication ID, Title, DOI, Year, Abstract (normalized), Publication Type, Ranking score and Method (keyword and/or semantic). This can be changed inside the script. The total number of records returned can be changed by using POOL_SIZE (see *Setup and run*).        

Most warnings can be safely ignored as long as the script finishes without crashing.   

### Known issues     

* Different versions of python elasticsearch module can cause errors like: TypeError: index() got an unexpected keyword argument 'document'.  
Using elasticsearch-7.16.3 should work (required for ES 6).
* Encoding errors (Linux only?) can be handled by running this script as: 
PYTHONIOENCODING=utf-8 python main.py

### Todo    

* Use [Specter2 Adapters](https://huggingface.co/allenai/specter2_base), to get slightly higher accuracy in the semantic searches. Adapters has been trained on the [SciRepEval dataset](https://huggingface.co/datasets/allenai/scirepeval), but does not currently work with Python 3.13+ and (or?) MacOS (python *adapters* library incompability).    
* Add more query filters  
