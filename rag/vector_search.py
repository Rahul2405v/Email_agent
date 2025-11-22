from .db import collection
from .config import EMBED_DIM

def vector_search(query_vec, k):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": 200,
                "limit": k
            }
        },
        {"$project": {"chunk": 1, "email_id": 1, "_id": 0}}
    ]
    return list(collection.aggregate(pipeline))
