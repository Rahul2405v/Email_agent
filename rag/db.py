from pymongo import MongoClient
from pymongo.collection import Collection
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection: Collection = db[COLLECTION_NAME]
