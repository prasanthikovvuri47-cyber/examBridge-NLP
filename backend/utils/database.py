# backend/utils/database.py
import os
from pymongo import MongoClient

# Use environment variable for MongoDB Atlas or fall back to local
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

try:
    # Set a short timeout for the initial connection attempt
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info() # Test connection
except Exception as e:
    print(f"⚠️ MongoDB Connection Error: {e}")
    print("Backend will continue running without persistence.")
    # Fallback to local (even if it 404s, we want the app to start)
    client = MongoClient(MONGO_URI)

db = client["exambridge"]
users_collection = db["users"]