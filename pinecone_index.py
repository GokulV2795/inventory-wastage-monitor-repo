import os
from dotenv import load_dotenv
load_dotenv()
import pinecone

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'inventory-wastage-index')

def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        # default dimension; update to match your embedding model
        pinecone.create_index(INDEX_NAME, dimension=1536)
    return pinecone.Index(INDEX_NAME)

def upsert_chunks(items):
    """items: list of dicts {'id': 'uuid', 'vector': [...], 'metadata': {...}}"""
    idx = init_pinecone()
    idx.upsert([(it['id'], it['vector'], it.get('metadata', {})) for it in items])

def query_similar(vector, top_k=5):
    idx = init_pinecone()
    resp = idx.query(vector=vector, top_k=top_k, include_metadata=True)
    return resp['matches']