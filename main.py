import os
import re
import tempfile
import requests
import pdfplumber
import logging
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ExamBridge AI API")

# ===============================
# CONFIGURATION
# ===============================
PDF_FOLDER = "gate_pdfs"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    if not YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY is not set.")
        return None
    try:
        return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        logger.error(f"YouTube API Error: {e}")
        return None

# Load Semantic Model
logger.info("Loading NLP model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Model loaded.")

templates = Jinja2Templates(directory="templates")

# ===============================
# UTILITY FUNCTIONS
# ===============================

def extract_pdf_text(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise ValueError(f"PDF extraction failed: {e}")
    return text

def extract_topics(text):
    if not text:
        return []
    lines = text.split("\n")
    topics = [line.strip() for line in lines if 10 < len(line.strip()) < 150 and not re.search(r'page \d+', line, re.I)]
    return list(dict.fromkeys(topics))

def compute_overall_similarity(college_text, gate_text):
    if not college_text or not gate_text:
        return 0.0
    embeddings = model.encode([college_text, gate_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity) * 100

def topic_wise_similarity(college_topics, gate_topics):
    if not college_topics or not gate_topics:
        return []
    college_embeddings = model.encode(college_topics, convert_to_tensor=True)
    gate_embeddings = model.encode(gate_topics, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(gate_embeddings, college_embeddings)
    results = []
    for i, gate_topic in enumerate(gate_topics):
        best_index = similarity_matrix[i].argmax()
        best_score = float(similarity_matrix[i][best_index]) * 100
        matched_topic = college_topics[best_index]
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
    results.sort(key=lambda x: (x["priority"] != "🚨 High", x["priority"] != "🟡 Medium", x["similarity"]))
    return results

def get_best_video_link(topic):
    youtube = get_youtube_client()
    if not youtube: return None
    try:
        search = youtube.search().list(
            q=f"GATE {topic} lecture technical full",
            part="snippet", type="video", videoDuration="long", 
            order="viewCount", maxResults=5
        ).execute()
        items = search.get('items', [])
        if not items:
            return None
        video_id = items[0]['id']['videoId']
        return f"https://www.youtube.com/watch?v={video_id}"
    except Exception as e:
        logger.error(f"Error fetching YouTube: {e}")
        return None

# ===============================
# ROUTES
# ===============================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    branches = []
    if os.path.exists(PDF_FOLDER):
        branches = [f.replace(".pdf", "") for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    return templates.TemplateResponse("index.html", {"request": request, "branches": branches})

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/analyze")
async def analyze(
    request: Request,
    syllabus_type: str = Form(...),
    selected_branch: str = Form(...),
    college_pdf: UploadFile = File(None),
    pasted_text: str = Form(None),
    syllabus_url: str = Form(None)
):
    logger.info(f"Analysis request received for branch: {selected_branch}")
    college_text = ""
    # Handle PDF Upload
    if syllabus_type == "Upload PDF" and college_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await college_pdf.read()
            tmp.write(content)
            college_text = extract_pdf_text(tmp.name)
        os.remove(tmp.name)
    # Handle Pasted Text
    elif syllabus_type == "Paste Text":
        college_text = pasted_text
    # Handle URL
    elif syllabus_type == "PDF Link" and syllabus_url:
        try:
            response = requests.get(syllabus_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                college_text = extract_pdf_text(tmp.name)
            os.remove(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF URL: {str(e)}")

    if not college_text:
        raise HTTPException(status_code=400, detail="No syllabus content found")

    # Load GATE Syllabus
    gate_pdf_path = os.path.join(PDF_FOLDER, f"{selected_branch}.pdf")
    if not os.path.exists(gate_pdf_path):
        raise HTTPException(status_code=404, detail="GATE branch file not found")
    
    gate_text = extract_pdf_text(gate_pdf_path)
    
    # Analyze
    overall_similarity = compute_overall_similarity(college_text, gate_text)
    college_topics = extract_topics(college_text)
    gate_topics = extract_topics(gate_text)
    results = topic_wise_similarity(college_topics, gate_topics)

    # Get video links for top gaps
    high_priority_gaps = [r for r in results if "High" in r["priority"]][:8]
    youtube_links = []
    for gap in high_priority_gaps:
        link = get_best_video_link(gap["gate_topic"])
        if link:
            youtube_links.append(link)

    comparison_summary = f"Match: {round(overall_similarity, 1)}%. Gaps: {len(high_priority_gaps)}."

    return {
        "comparison_result": comparison_summary,
        "youtube_links": youtube_links,
        "overall_similarity": round(overall_similarity, 1),
        "results": results,
        "gate_topic_count": len(gate_topics),
        "critical_gaps": len(high_priority_gaps)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
