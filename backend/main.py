from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.services.nlp_service import extract_text_from_pdf, extract_topics, compute_overall_similarity, topic_wise_similarity_ranking, get_model, get_util
from backend.services.youtube_service import fetch_youtube_videos, get_video_summary
import os
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ExamBridge AI API")

# -------------------------------
# CORS CONFIG - Allow your GitHub Pages site
# -------------------------------
origins = [
    "https://pothulaannapurna8.github.io",
    "https://exam-bridge-nexus.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "*" # Allowed for debugging, narrow down in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_FOLDER = "gate_pdfs"

@app.get("/")
def home():
    logger.info("Root endpoint hit.")
    return {"status": "ExamBridge AI API is Running 🚀"}

@app.get("/health")
def health_check():
    logger.info("Health check endpoint hit.")
    return {"status": "running"}

@app.post("/analyze/{branch}")
async def analyze(branch: str, file: UploadFile = File(...)):
    """
    Main endpoint for the GitHub frontend.
    1. Extracts text from uploaded PDF.
    2. Compares against the specified GATE branch PDF.
    3. Returns similarity score, gaps, and summarized YouTube lectures.
    """
    logger.info(f"Received analysis request for branch: {branch}")
    
    # 1. Extract College Syllabus Text
    try:
        content = await file.read()
        logger.info(f"File '{file.filename}' read successfully. Size: {len(content)} bytes.")
        college_text = extract_text_from_pdf(content)
        
        if not college_text.strip():
            logger.warning("Empty text extracted from uploaded PDF.")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
    except Exception as e:
        logger.error(f"Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

    # 2. Load GATE Syllabus
    gate_pdf_path = os.path.join(PDF_FOLDER, f"{branch}.pdf")
    if not os.path.exists(gate_pdf_path):
        # Try root as fallback
        gate_pdf_path = f"{branch}.pdf"
        
    if not os.path.exists(gate_pdf_path):
        logger.error(f"GATE branch file not found: {gate_pdf_path}")
        raise HTTPException(status_code=404, detail=f"GATE branch {branch} not found")
        
    try:
        with open(gate_pdf_path, "rb") as f:
            gate_content = f.read()
            gate_text = extract_text_from_pdf(gate_content)
            logger.info(f"GATE syllabus for '{branch}' loaded and extracted.")
    except Exception as e:
        logger.error(f"Error reading GATE syllabus: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read GATE syllabus: {str(e)}")

    # 3. Perform AI Analysis
    logger.info("Starting AI NLP analysis...")
    try:
        overall_similarity = compute_overall_similarity(college_text, gate_text)
        college_topics = extract_topics(college_text)
        gate_topics = extract_topics(gate_text)
        results = topic_wise_similarity_ranking(college_topics, gate_topics)
    except Exception as e:
        logger.error(f"AI Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

    # 4. Get Enriched Recommendations (Gaps + Summaries)
    high_priority_gaps = [r for r in results if "High" in r["priority"]][:8]
    recommendations = []
    youtube_links = []
    
    logger.info(f"Found {len(high_priority_gaps)} high priority gaps. Fetching recommendations...")
    
    try:
        model = get_model()
        util = get_util()
    except Exception as e:
        logger.error(f"Failed to load NLP model for summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize NLP model")

    for gap in high_priority_gaps:
        topic_name = gap["gate_topic"]
        videos = fetch_youtube_videos(topic_name)
        
        if videos and isinstance(videos[0], dict) and "error" not in videos[0]:
            top_video = videos[0]
            youtube_links.append(top_video['url'])
            
            # Add AI Summary
            v_id = top_video['url'].split("v=")[-1]
            top_video['summary'] = get_video_summary(v_id, topic_name, model, util)
            
            recommendations.append({
                "topic": topic_name,
                "video": top_video
            })
        elif videos and isinstance(videos[0], dict) and "error" in videos[0]:
            logger.warning(f"Error fetching YouTube videos for {topic_name}: {videos[0]['error']}")

    comparison_summary = f"Your syllabus has an overall match of {round(overall_similarity, 1)}% with the GATE {branch} syllabus. We identified {len(high_priority_gaps)} critical gaps where topics are either missing or have low similarity."

    logger.info("Analysis complete. Returning response.")
    
    # Return structured JSON as requested
    return {
        "comparison_result": comparison_summary,
        "youtube_links": youtube_links,
        # Keep original fields for backward compatibility/debugging if needed by current frontend
        "overall_similarity": round(overall_similarity, 1),
        "critical_gaps": len(high_priority_gaps),
        "gate_topic_count": len(gate_topics),
        "results": results,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
