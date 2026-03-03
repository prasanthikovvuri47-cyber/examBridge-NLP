from backend.routes.auth import router as auth_router
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.services.nlp_service import extract_text_from_pdf, analyze_text_against_syllabus
from backend.services.youtube_service import fetch_youtube_videos
from backend.utils.syllabus_loader import load_syllabus
import os
import uvicorn

app = FastAPI(title="ExamBridge Nexus Backend")

# -------------------------------
# CORS CONFIG
# -------------------------------
origins = [
    "https://pothulaannapurna8.github.io",
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# ROUTERS
# -------------------------------
app.include_router(auth_router)

# -------------------------------
# TEMP STORAGE (Upgrade to DB later)
# -------------------------------
analysis_cache = {}
user_progress = {}

# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "ExamBridge Nexus Backend Running 🚀",
        "status": "healthy"
    }

# -------------------------------
# ANALYZE PDF
# -------------------------------
@app.post("/analyze/{branch}")
async def analyze_pdf(branch: str, file: UploadFile = File(...)):
    branch = branch.lower()

    syllabus_data = load_syllabus(branch)
    if syllabus_data is None:
        raise HTTPException(status_code=400, detail="Branch not supported")

    contents = await file.read()
    # Handle both bytes and strings if needed (nlp_service expects bytes for extraction)
    text = extract_text_from_pdf(contents)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text")

    results = analyze_text_against_syllabus(text, syllabus_data)

    analysis_cache[branch] = results

    return {
        "success": True,
        "branch": branch,
        "analysis": results
    }

# -------------------------------
# DASHBOARD (Chart.js Friendly)
# -------------------------------
@app.get("/dashboard/{branch}")
def get_dashboard(branch: str):
    branch = branch.lower()

    if branch not in analysis_cache:
        raise HTTPException(status_code=404, detail="No analysis found")

    analysis = analysis_cache[branch]

    subjects = []
    scores = []

    for subject, topics in analysis.items():
        avg_score = sum(topics.values()) / len(topics)
        subjects.append(subject)
        scores.append(round(avg_score, 2))

    return {
        "success": True,
        "labels": subjects,
        "data": scores
    }

# -------------------------------
# PRIORITY TOPICS
# -------------------------------
@app.get("/priority/{branch}")
def get_priority(branch: str, top_n: int = 5):
    branch = branch.lower()

    if branch not in analysis_cache:
        raise HTTPException(status_code=404, detail="No analysis found")

    analysis = analysis_cache[branch]

    all_scores = []

    for subject, topics in analysis.items():
        for topic, score in topics.items():
            all_scores.append((subject, topic, score))

    ranked = sorted(all_scores, key=lambda x: x[2], reverse=True)
    top_topics = ranked[:top_n]

    enriched_topics = []

    for subject, topic, score in top_topics:
        videos = fetch_youtube_videos(f"GATE {topic} lecture")

        enriched_topics.append({
            "subject": subject,
            "topic": topic,
            "score": score,
            "videos": videos
        })

    return {
        "success": True,
        "branch": branch,
        "priority_topics": enriched_topics
    }

# -------------------------------
# STUDY PLAN DOWNLOAD
# -------------------------------
@app.get("/study-plan/{branch}")
def generate_study_plan(branch: str):
    branch = branch.lower()

    if branch not in analysis_cache:
        raise HTTPException(status_code=404, detail="No analysis found")

    analysis = analysis_cache[branch]
    plan = []

    for subject, topics in analysis.items():
        weak_topics = [t for t, s in topics.items() if s < 50]
        if weak_topics:
            plan.append({
                "subject": subject,
                "focus_topics": weak_topics
            })

    return JSONResponse(content={
        "success": True,
        "branch": branch,
        "study_plan": plan
    })

# -------------------------------
# MARK TOPIC AS DONE
# -------------------------------
@app.post("/mark-done")
def mark_done(data: dict = Body(...)):
    email = data.get("email")
    topic = data.get("topic")

    if not email or not topic:
        raise HTTPException(status_code=400, detail="Missing data")

    if email not in user_progress:
        user_progress[email] = []

    if topic not in user_progress[email]:
        user_progress[email].append(topic)

    return {"success": True}

# -------------------------------
# GET USER PROGRESS
# -------------------------------
@app.get("/progress/{email}")
def get_progress(email: str):
    return {
        "success": True,
        "completed_topics": user_progress.get(email, [])
    }

# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)