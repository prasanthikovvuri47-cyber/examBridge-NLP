import pdfplumber
import io
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy load heavy modules to prevent startup issues
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    return _model

def get_util():
    try:
        from sentence_transformers import util
        return util
    except Exception as e:
        logger.error(f"Failed to load sentence_transformers.util: {e}")
        raise RuntimeError(f"Utility loading failed: {e}")

def extract_text_from_pdf(contents: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise ValueError(f"PDF extraction failed: {e}")
    return text

def extract_topics(text):
    if not text:
        return []
    # Clean text and split by newlines
    lines = text.split("\n")
    # Filter lines that look like topics (length check + no page numbers)
    topics = [line.strip() for line in lines if 10 < len(line.strip()) < 150 and not re.search(r'page \d+', line, re.I)]
    # Deduplicate while preserving order
    return list(dict.fromkeys(topics))

def compute_overall_similarity(text1, text2):
    logger.info("Computing overall similarity...")
    model = get_model()
    util = get_util()
    embeddings = model.encode([text1, text2])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    score = float(similarity[0][0]) * 100
    logger.info(f"Overall similarity computed: {score:.2f}%")
    return score

def topic_wise_similarity_ranking(college_topics, gate_topics):
    if not college_topics or not gate_topics:
        logger.warning("Empty topics provided for comparison.")
        return []
        
    logger.info(f"Starting topic-wise comparison (GATE topics: {len(gate_topics)}, College topics: {len(college_topics)})")
    model = get_model()
    util = get_util()
    
    # Batch encode for performance
    college_embeddings = model.encode(college_topics, convert_to_tensor=True)
    gate_embeddings = model.encode(gate_topics, convert_to_tensor=True)
    
    # Compute similarity matrix
    similarity_matrix = util.cos_sim(gate_embeddings, college_embeddings)
    
    results = []
    for i, gate_topic in enumerate(gate_topics):
        # Find best match in college syllabus
        best_index = similarity_matrix[i].argmax()
        best_score = float(similarity_matrix[i][best_index]) * 100
        matched_topic = college_topics[best_index]
        
        # Priority logic
        if best_score < 40:
            priority = "🚨 High"
        elif 40 <= best_score < 70:
            priority = "🟡 Medium"
        else:
            priority = "✅ Low"
            
        results.append({
            "gate_topic": gate_topic,
            "matched_topic": matched_topic,
            "similarity": round(best_score, 1),
            "priority": priority
        })
        
    # Sort by priority and then similarity (Critical gaps first)
    results.sort(key=lambda x: (x["priority"] != "🚨 High", x["priority"] != "🟡 Medium", x["similarity"]))
    logger.info("Topic-wise comparison completed.")
    return results
