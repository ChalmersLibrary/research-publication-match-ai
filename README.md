# research-publication-match-ai

App for locating publications within specific subject(s) and/or matching other criteria in [Chalmers Research CRIS](https://research.chalmers.se), by querying both ES for keyword hits and vector store for semantic hits — and then merge the results with combined rankings (reciprocal rank fusion).     
Uses [FAISS](https://github.com/facebookresearch/faiss) and the [allenai/specter2](https://huggingface.co/allenai/specter2) model for embedding.    

### Requirements  

* Read access to an elastic research-publications index.
* Python 3.x with required modules
    - FAISS, Sentence-transformers, Elasticsearch (<=7.16.3), Peft

### Setup   

- Install python libraries    
``pip install faiss-cpu sentence-transformers python-dotenv peft os csv re``    

- Install working library for Elasticsearch 6.x (for compability)    
``pip install elasticsearch-7.16.3``    

- Create local vector store (FAISS and jsonl indexes) for semantic search (may take a while, must be re-run if the ES index content change)       
``python build_index.py``

- Edit *main.py* and add relevant keywords in the **QUERY** param

- Create an **.env** file with local settings, using *env_example* as template:    
    - ES_URL - elasticsearch base URL   
    - ES_UID - elasticsearch User ID   
    - ES_PW - elasticsearch password   
    - ES_INDEX - name of the ES index (eg. *research-publications-static-20260101*)   
    - OUTFILE_CSV - name of output CSV file (default: results)    
    - QUERY - query string (keywords) to be used, eg. *maritime marine shipping seafood aquaculture*
    - FETCH_FIELDS - fields that should be retrieved from Chalmers CRIS, eg. *Id,Title,IdentifierDoi[0],Year,PublicationType.NameEng*   
    - START_YEAR - only include publications from this year forwards (default: *2014*)        

- Run the script    
``python main.py``   

### Output  

The output is written to a local CSV file in the current directory (see *main.py* for details and adjust if needed). File name is specified in the *.env* file (default: results.YYYYMMDD.hhmmss.csv).   

The current (proof of concept) version only return the top 50 hits, with Publication ID, Title, DOI, Year, Publication Type, Ranking score and Method (keyword and/or semantic). This can be changed inside the script.   

### Known issues     

* Different versions of python elasticsearch module can cause errors like: TypeError: index() got an unexpected keyword argument 'document'.  
Using elasticsearch-7.16.3 should work (required for ES 6).
* Encoding errors (Linux only?) can be handled by running this script as: 
PYTHONIOENCODING=utf-8 python main.py
