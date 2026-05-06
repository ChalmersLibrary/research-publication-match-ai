# research-publication-match-ai

App for locating publications within specific subject(s) and/or matching other criteria in [Chalmers Research CRIS](https://research.chalmers.se), by querying both ES for keyword hits and vector store for semantic hits — and then merge the results with combined rankings (reciprocal rank fusion).     
Uses [FAISS](https://github.com/facebookresearch/faiss) and the [allenai/specter2](https://huggingface.co/allenai/specter2) model for embedding.    

### Requirements  

* Read access to an elasticsearch Chalmers research-publications index (please use a static index if possible, preferred both for indexing and performance)   
* Python 3.x with required modules
    - FAISS, Sentence-transformers, Elasticsearch (<=7.16.3), Peft

### Setup   

- Install python libraries    
``pip install faiss-cpu sentence-transformers python-dotenv peft os csv re``    

- Install working library for Elasticsearch 6.x (for compability)    
``pip install elasticsearch-7.16.3``    

- Create local vector store (FAISS and jsonl indexes) for semantic search (may take a while, must be re-run if the ES index content change, but not if just modifying the query or search filters)       
``python build_index.py``   

- Create an **.env** file with local settings, using *env_example* as template (note the format of the examples):    
    - ES_URL - elasticsearch base URL   
    - ES_UID - elasticsearch user ID   
    - ES_PW - elasticsearch password   
    - ES_INDEX - name of the elastic index (eg. *research-publications-static-20260101*)   
    - OUTFILE_CSV - name of output CSV file (default: *results*)    
    - QUERY - query string (keywords, comma separated) to be used, eg. *maritime marine shipping seafood aquaculture blue bioeconomy ocean currents*
    - FETCH_FIELDS - fields that should be retrieved from Chalmers CRIS, eg. *Id,Title,IdentifierDoi[0],Year,PublicationType.NameEng*   
    - START_YEAR - only include publications from this year forwards (default: *2014*)        

- Run the script    
``python main.py``   

### Output  

The output is written to a local CSV file in the current directory (see *main.py* for details and adjust if needed). File name is specified in the *.env* file (default: results.YYYYMMDD.hhmmss.csv).   

The current (proof of concept) version only return the top 50 hits, with Publication ID, Title, DOI, Year, Publication Type, Ranking score and Method (keyword and/or semantic). This can be changed inside the script.   

Most warnings can be safely ignored as long as the script finishes without crashing.   

### Known issues     

* Different versions of python elasticsearch module can cause errors like: TypeError: index() got an unexpected keyword argument 'document'.  
Using elasticsearch-7.16.3 should work (required for ES 6).
* Encoding errors (Linux only?) can be handled by running this script as: 
PYTHONIOENCODING=utf-8 python main.py
