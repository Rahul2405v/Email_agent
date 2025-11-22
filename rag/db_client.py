import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
_client = MongoClient(MONGO_URI)
_db = _client["rag_db"]

_prompts_col = _db["prompts"]
_emails_col = _db["mock_emails"]


def get_prompts():
    doc = _prompts_col.find_one({"_id": "prompts"})
    if doc and "data" in doc:
        return doc["data"]

    # fallback: load from local file and store in DB
    if os.path.exists("prompts.json"):
        with open("prompts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        save_prompts(data)
        return data

    return {}


def save_prompts(data: dict):
    _prompts_col.update_one({"_id": "prompts"}, {"$set": {"data": data}}, upsert=True)
    return True


def get_emails():
    docs = list(_emails_col.find({}, {"_id": 0}))
    if docs:
        return docs

    # fallback: seed from file if present
    if os.path.exists("mock_emails.json"):
        with open("mock_emails.json", "r", encoding="utf-8") as f:
            items = json.load(f)
        if items:
            # replace any existing docs
            _emails_col.delete_many({})
            # Ensure inserted docs have no _id conflicts
            _emails_col.insert_many(items)
            return list(_emails_col.find({}, {"_id": 0}))

    return []


def update_email(email_id: str, update_fields: dict):
    _emails_col.update_one({"id": email_id}, {"$set": update_fields}, upsert=True)


def replace_all_emails(emails_list: list):
    _emails_col.delete_many({})
    if emails_list:
        _emails_col.insert_many(emails_list)
    return True
