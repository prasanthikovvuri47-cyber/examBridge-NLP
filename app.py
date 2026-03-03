import streamlit as st

# MUST BE THE FIRST ST-COMMAND
st.set_page_config(
    page_title="ExamBridge AI | GATE Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pdfplumber
import os
import re
import sys
import numpy as np
import requests
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi

# Environment Guard Removed as requested to allow deployment on Render's Python environment

# Lazy load heavy modules
@st.cache_resource
def get_transformer_modules():
    from sentence_transformers import SentenceTransformer, util
    return SentenceTransformer, util

@st.cache_resource
def get_model():
    SentenceTransformer, _ = get_transformer_modules()
    return SentenceTransformer('all-MiniLM-L6-v2')

from googleapiclient.discovery import build

# ===============================
# CONFIGURATION
# ===============================

PDF_FOLDER = "gate_pdfs"
# Check for secrets securely
# On Hugging Face, 'Secrets' are stored as environment variables.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Fallback for local testing if env var is missing
if not YOUTUBE_API_KEY:
    try:
        # We only access st.secrets if we think it might exist to avoid visual warnings on some platforms
        if os.path.exists(".streamlit/secrets.toml") or os.path.exists(os.path.expanduser("~/.streamlit/secrets.toml")):
            YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY")
    except Exception:
        pass

# Final fallback to hardcoded key if all else fails
if not YOUTUBE_API_KEY:
    YOUTUBE_API_KEY = "AIzaSyAsJzyUy_IaAglkSUBYVXZUjxH1ehLG8b0"
DEPLOYMENT_MODE = False  # False = Dev Mode (show views & likes)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .topic-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .video-card {
        background: white;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
        border: 1px solid #eee;
    }
    .video-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .video-info {
        padding: 12px;
    }
    .video-title {
        font-size: 0.95rem;
        font-weight: bold;
        color: #1E1E1E;
        line-height: 1.3;
        margin-bottom: 5px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .video-meta {
        font-size: 0.8rem;
        color: #666;
    }
    .summary-card {
        background-color: #fff5f5;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        border-left: 3px solid #FF4B4B;
        font-size: 0.85rem;
        color: #444;
    }
    .summary-title {
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }
    h1, h2, h3 {
        color: #1E1E1E;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🎓")
with col2:
    st.title("ExamBridge AI")
    st.markdown("### Semantic Syllabus Gap Detection System")
    st.caption("Powered by Sentence Transformers & YouTube Search API")

st.divider()

# Model loading handled lazily in analysis section

try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    st.warning(f"YouTube API Error: {e}")
    youtube = None

# ===============================
# UTILITY FUNCTIONS
# ===============================

def extract_pdf_text(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
    return text


def extract_topics(text):
    if not text:
        return []
    lines = text.split("\n")
    # Improved topic extraction filtering
    topics = [line.strip() for line in lines if 10 < len(line.strip()) < 150 and not re.search(r'page \d+', line, re.I)]
    return list(dict.fromkeys(topics)) # Maintain order, remove duplicates


def compute_overall_similarity(college_text, gate_text):
    if not college_text or not gate_text:
        return 0.0
    _, util = get_transformer_modules()
    model = get_model()
    embeddings = model.encode([college_text, gate_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity) * 100


def topic_wise_similarity(college_topics, gate_topics):
    if not college_topics or not gate_topics:
        return []

    _, util = get_transformer_modules()
    model = get_model()
    college_embeddings = model.encode(college_topics, convert_to_tensor=True)
    gate_embeddings = model.encode(gate_topics, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(gate_embeddings, college_embeddings)

    results = []

    for i, gate_topic in enumerate(gate_topics):
        best_index = similarity_matrix[i].argmax()
        best_score = float(similarity_matrix[i][best_index]) * 100
        matched_topic = college_topics[best_index]

        if best_score < 40: # Lowered threshold slightly for more "High" priority gaps
            priority = "🚨 High"
        elif 40 <= best_score < 70:
            priority = "🟡 Medium"
        else:
            priority = "✅ Low"

        results.append({
            "gate_topic": gate_topic,
            "matched_topic": matched_topic,
            "similarity": best_score,
            "priority": priority
        })

    # Sort by priority and similarity
    results.sort(key=lambda x: (
        x["priority"] != "🚨 High",
        x["priority"] != "🟡 Medium",
        x["similarity"]
    ))

    return results


def parse_duration(duration_str):
    """
    Parses ISO 8601 duration string (e.g., PT15M33S) into total seconds.
    """
    import re
    if not duration_str:
        return 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds


def get_best_video_link(topic):
    if not youtube:
        return None
    try:
        # Request 'long' videos (> 20 mins) ordered by viewCount to get the most popular lectures
        search = youtube.search().list(
            q=f"GATE {topic} lecture technical full",
            part="snippet",
            type="video",
            videoDuration="long", 
            order="viewCount",  # Priority: Max views
            maxResults=10
        ).execute()

        video_ids = [item['id']['videoId'] for item in search['items']]
        
        # If no 'long' videos, try 'medium' (4-20 mins) also ordered by views
        if not video_ids:
            search = youtube.search().list(
                q=f"GATE {topic} lecture technical",
                part="snippet",
                type="video",
                videoDuration="medium",
                order="viewCount",
                maxResults=10
            ).execute()
            video_ids = [item['id']['videoId'] for item in search['items']]

        if not video_ids:
            return None

        # Fetch detailed stats for the top results found
        videos_data = youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=",".join(video_ids)
        ).execute()

        best_video = None
        best_score = -1

        for item in videos_data['items']:
            duration_sec = parse_duration(item['contentDetails'].get('duration', ''))
            
            # Strict filtering: exclude anything that isn't a substantial lecture
            if duration_sec < 300: # Increased to 5 minutes to be safe
                continue
                
            views = int(item['statistics'].get('viewCount', 0))
            likes = int(item['statistics'].get('likeCount', 0))
            
            # Simple but effective score: Views + Likes weighted heavily
            score = views + (likes * 50) 

            if score > best_score:
                best_score = score
                best_video = {
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "title": item['snippet']['title'],
                    "thumbnail": item['snippet']['thumbnails']['high']['url'],
                    "duration": item['contentDetails']['duration'].replace('PT', '').lower(),
                    "channel": item['snippet']['channelTitle']
                }

        return best_video
    except Exception as e:
        return None


def get_video_summary(video_id, topic):
    """
    Generates a memory-efficient extractive summary by ranking transcript sentences 
    relative to the target topic.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t['text'] for t in transcript_list])
        
        # Split into sentences (simple split for speed/memory)
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', full_text) if len(s.strip()) > 20]
        
        if not sentences:
            return "Transcript available but no substantial content found for summary."

        # Get models
        _, util = get_transformer_modules()
        model = get_model()
        
        # Encode topic and sentences
        topic_emb = model.encode(topic, convert_to_tensor=True)
        sentence_embs = model.encode(sentences, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(topic_emb, sentence_embs)[0]
        
        # Get top 3 most relevant sentences
        top_indices = similarities.argsort(descending=True)[:3]
        top_sentences = [sentences[idx] for idx in top_indices.tolist()]
        
        # Format summary
        summary = " • " + "\n • ".join(top_sentences)
        return summary
    except Exception:
        return "Summary not available (transcripts might be disabled for this video)."


# ===============================
# USER INTERFACE - INPUTS
# ===============================

input_col, extra_col = st.columns([2, 1])

with input_col:
    st.subheader("📂 College Syllabus")
    college_upload_method = st.radio(
        "Input Method:",
        ["Upload PDF", "Paste Text", "PDF Link"],
        horizontal=True
    )

    college_text = ""
    if college_upload_method == "Upload PDF":
        college_pdf = st.file_uploader("Upload your syllabus PDF", type="pdf")
        if college_pdf:
            with st.spinner("Extracting text from PDF..."):
                college_text = extract_pdf_text(college_pdf)
    
    elif college_upload_method == "Paste Text":
        college_text = st.text_area("Paste your college syllabus content here:", height=200, placeholder="Unit 1: Data Structures, Unit 2: Algorithms...")
    
    else:
        college_url = st.text_input("Enter Syllabus URL:", placeholder="https://example.com/syllabus.pdf")
        if college_url:
            try:
                with st.spinner("Fetching and reading PDF link..."):
                    response = requests.get(college_url)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(response.content)
                        college_text = extract_pdf_text(tmp.name)
                        os.remove(tmp.name)
            except Exception as e:
                st.error(f"Failed to fetch PDF: {e}")

with extra_col:
    st.subheader("🎯 Target GATE")
    # Check both PDF_FOLDER and root directory for flexibility
    potential_paths = [PDF_FOLDER, "."]
    branches = []
    
    for path in potential_paths:
        if os.path.exists(path):
            branches.extend([
                file.replace(".pdf", "")
                for file in os.listdir(path)
                if file.endswith(".pdf")
            ])
    
    # Remove duplicates if any
    branches = list(set(branches))
    selected_branch = st.selectbox("Select GATE Branch:", sorted(branches))
    
    st.info("💡 **Pro Tip:** Pasting text is often faster for quick checks!")

# ===============================
# ANALYSIS EXECUTION
# ===============================

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run Semantic Analysis"):

    if not college_text:
        st.warning("Please provide your college syllabus first.")
    elif not selected_branch:
        st.warning("Please select a target GATE branch.")
    else:
        # Check for syllabus in both folder and root
        gate_pdf_path = os.path.join(PDF_FOLDER, f"{selected_branch}.pdf")
        if not os.path.exists(gate_pdf_path):
            gate_pdf_path = f"{selected_branch}.pdf" # Try root
            
        if not os.path.exists(gate_pdf_path):
            st.error(f"GATE syllabus file '{selected_branch}.pdf' not found in {PDF_FOLDER} or root.")
        else:
            with st.spinner("🔬 Performing Deep Semantic Analysis..."):
                # Load GATE Syllabus
                with open(gate_pdf_path, "rb") as f:
                    gate_text = extract_pdf_text(f)
                
                # Run Comparison
                overall_similarity = compute_overall_similarity(college_text, gate_text)
                college_topics = extract_topics(college_text)
                gate_topics = extract_topics(gate_text)
                results = topic_wise_similarity(college_topics, gate_topics)

            # ===============================
            # RESULTS DISPLAY
            # ===============================
            
            st.header("📊 Analysis Results")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Overall Match", f"{overall_similarity:.1f}%", delta=None)
            with m2:
                # Use a more robust check to avoid linter confusion
                high_priority_count = sum(1 for r in results if str(r.get("priority", "")).endswith("High"))
                st.metric("Critical Gaps", high_priority_count, delta=f"{high_priority_count} topics", delta_color="inverse")
            with m3:
                st.metric("Total Topics", len(gate_topics))

            st.divider()

            # Detailed Results Split
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                st.subheader("🔎 Topic Mapping & Priority")
                display_df = [{
                    "GATE Topic": r["gate_topic"],
                    "Matching College Topic": r["matched_topic"],
                    "Confidence (%)": f"{r['similarity']:.1f}",
                    "Status": r["priority"]
                } for r in results]
                st.dataframe(display_df, use_container_width=True, height=500)

            with res_col2:
                st.subheader("🔥 Learning Bridge")
                st.write("Recommended lectures for High Priority gaps:")
                
                # Filter results explicitly for strings containing "High"
                high_priority_results = [r for r in results if isinstance(r.get("priority"), str) and "High" in str(r.get("priority"))]
                
                if not high_priority_results:
                    st.success("Great job! Your syllabus covers the GATE topics well.")
                else:
                    for i in range(min(8, len(high_priority_results))):
                        r = high_priority_results[i]
                        topic_name = str(r.get("gate_topic", "Unknown Topic"))
                        
                        video_data = get_best_video_link(topic_name)
                        
                        if video_data:
                            with st.container():
                                st.markdown(f"""
                                <div class="video-card">
                                    <img src="{video_data['thumbnail']}" style="width:100%; height:auto;">
                                    <div class="video-info">
                                        <div class="video-title">{video_data['title']}</div>
                                        <div class="video-meta">📺 {video_data['channel']} • ⏳ {video_data['duration']}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.video(video_data['url'])
                                
                                # Fetch and show summary
                                with st.expander("📝 Key Lecture Points (AI Summary)"):
                                    v_id = video_data['url'].split("v=")[-1]
                                    with st.spinner("Summarizing lecture..."):
                                        summary = get_video_summary(v_id, topic_name)
                                        st.markdown(f'<div class="summary-card"><div class="summary-title">Relevant Highlights:</div>{summary}</div>', unsafe_allow_html=True)
                        else:
                            with st.expander(f"📌 {topic_name}"):
                                st.write("Searching for lectures...")
                                st.markdown(f"[Search on YouTube](https://www.youtube.com/results?search_query=GATE+{topic_name.replace(' ', '+')}+lecture)")

            st.balloons()
