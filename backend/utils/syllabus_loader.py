import os
import json

SYLLABUS_DIR = "backend/syllabus"

def load_syllabus(branch: str):
    file_path = os.path.join(SYLLABUS_DIR, f"{branch}.json")
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)