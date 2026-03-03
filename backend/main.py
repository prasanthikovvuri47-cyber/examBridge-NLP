from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.services.nlp_service import extract_text_from_pdf, extract_topics, compute_overall_similarity, topic_wise_similarity_ranking, get_model, get_util
from backend.services.youtube_service import fetch_youtube_videos, get_video_summary
import os
import uvicorn

app = FastAPI(title="ExamBridge AI API")

# -------------------------------
# CORS CONFIG - Allow your GitHub Pages site
# -------------------------------
origins = [
    "https://pothulaannapurna8.github.io",
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
    return {"status": "ExamBridge AI API is Running 🚀"}

@app.post("/analyze/{branch}")
async def analyze(branch: str, file: UploadFile = File(...)):
    """
    Main endpoint for the GitHub frontend.
    1. Extracts text from uploaded PDF.
    2. Compares against the specified GATE branch PDF.
    3. Returns similarity score, gaps, and summarized YouTube lectures.
    """
    # 1. Extract College Syllabus Text
    content = await file.read()
    college_text = extract_text_from_pdf(content)
    
    if not college_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # 2. Load GATE Syllabus
    gate_pdf_path = os.path.join(PDF_FOLDER, f"{branch}.pdf")
    if not os.path.exists(gate_pdf_path):
        # Try root as fallback
        gate_pdf_path = f"{branch}.pdf"
        
    if not os.path.exists(gate_pdf_path):
        raise HTTPException(status_code=404, detail=f"GATE branch {branch} not found")
        
    with open(gate_pdf_path, "rb") as f:
        gate_content = f.read()
        gate_text = extract_text_from_pdf(gate_content)

    # 3. Perform AI Analysis
    overall_similarity = compute_overall_similarity(college_text, gate_text)
    college_topics = extract_topics(college_text)
    gate_topics = extract_topics(gate_text)
    results = topic_wise_similarity_ranking(college_topics, gate_topics)

    # 4. Get Enriched Recommendations (Gaps + Summaries)
    high_priority_gaps = [r for r in results if "High" in r["priority"]][:8]
    recommendations = []
    
    model = get_model()
    util = get_util()

    for gap in high_priority_gaps:
        topic_name = gap["gate_topic"]
        videos = fetch_youtube_videos(topic_name)
        
        if videos and "error" not in videos[0]:
            top_video = videos[0]
            # Add AI Summary
            v_id = top_video['url'].split("v=")[-1]
            top_video['summary'] = get_video_summary(v_id, topic_name, model, util)
            
            recommendations.append({
                "topic": topic_name,
                "video": top_video
            })

    return {
        "overall_similarity": round(overall_similarity, 1),
        "critical_gaps": len(high_priority_gaps),
        "gate_topic_count": len(gate_topics),
        "results": results,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)