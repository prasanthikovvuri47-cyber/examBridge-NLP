import pdfplumber
import io
import re
from functools import lru_cache

# Lazy load heavy modules to prevent startup issues
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_util():
    from sentence_transformers import util
    return util

def extract_text_from_pdf(contents: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    model = get_model()
    util = get_util()
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity[0][0])

def analyze_text_against_syllabus(text: str, syllabus_data: dict):
    results = {}
    
    for subject, topics in syllabus_data.items():
        subject_scores = {}
        
        for topic_name, subtopics in topics.items():
            combined_topic_text = f"{topic_name}: " + " ".join(subtopics)
            score = calculate_similarity(text, combined_topic_text)
            subject_scores[topic_name] = round(score * 100, 1) # Convert to percentage

        sorted_scores = dict(
            sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)
        )

        results[subject] = sorted_scores

    return results
