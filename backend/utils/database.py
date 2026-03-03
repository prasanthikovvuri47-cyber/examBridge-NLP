# backend/utils/database.py
from pymongo import MongoClient

# Connect to MongoDB (adjust URI if needed)
client = MongoClient("mongodb://localhost:27017/")

# Select your database
db = client["exambridge"]

# Define collections
users_collection = db["users"]