import streamlit as st
import pdfplumber
import os
import re
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
import numpy as np
import requests
import tempfile

# ===============================
# CONFIGURATION
# ===============================

PDF_FOLDER = "gate_pdfs"
YOUTUBE_API_KEY = "AIzaSyAsJzyUy_IaAglkSUBYVXZUjxH1ehLG8b0"   # Add your API key here
DEPLOYMENT_MODE = True  # True = hide views & likes

st.set_page_config(page_title="ExamBridge AI", layout="wide")
st.title("🚀 ExamBridge AI - GATE Syllabus Analyzer")
st.caption("Semantic AI Based Syllabus Gap Detection System")

# ===============================
# LOAD SEMANTIC MODEL
# ===============================

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# ===============================
# UTILITY FUNCTIONS
# ===============================

def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


def extract_topics(text):
    lines = text.split("\n")
    topics = [line.strip() for line in lines if 10 < len(line.strip()) < 120]
    return list(set(topics))


def compute_overall_similarity(college_text, gate_text):
    embeddings = model.encode([college_text, gate_text])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity) * 100


def topic_wise_similarity(college_topics, gate_topics):

    college_embeddings = model.encode(college_topics, convert_to_tensor=True)
    gate_embeddings = model.encode(gate_topics, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(gate_embeddings, college_embeddings)

    results = []

    for i, gate_topic in enumerate(gate_topics):
        best_index = similarity_matrix[i].argmax()
        best_score = float(similarity_matrix[i][best_index]) * 100
        matched_topic = college_topics[best_index]

        if best_score < 50:
            priority = "High"
        elif 50 <= best_score < 75:
            priority = "Medium"
        else:
            priority = "Low"

        results.append({
            "gate_topic": gate_topic,
            "matched_topic": matched_topic,
            "similarity": best_score,
            "priority": priority
        })

    results.sort(key=lambda x: (
        x["priority"] != "High",
        x["priority"] != "Medium",
        -x["similarity"]
    ))

    return results


def get_best_video_link(topic):
    try:
        search = youtube.search().list(
            q=f"GATE {topic} full lecture",
            part="snippet",
            type="video",
            maxResults=5
        ).execute()

        video_ids = [item['id']['videoId'] for item in search['items']]

        videos_data = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids)
        ).execute()

        best_video = None
        best_score = 0

        for item in videos_data['items']:
            views = int(item['statistics'].get('viewCount', 0))
            likes = int(item['statistics'].get('likeCount', 0))
            score = views + (likes * 10)

            if score > best_score:
                best_score = score
                best_video = f"https://www.youtube.com/watch?v={item['id']}"

        return best_video
    except:
        return None


# ===============================
# USER INTERFACE
# ===============================

st.header("📂 Provide Your College Syllabus")
college_upload_method = st.radio("Choose College Syllabus Input Method:", ["Upload PDF File", "Provide PDF Link"])

college_pdf = None
college_url = ""
if college_upload_method == "Upload PDF File":
    college_pdf = st.file_uploader("Upload PDF", type="pdf")
else:
    college_url = st.text_input("Enter College Syllabus URL:")

st.header("🎓 Select GATE Syllabus")

if os.path.exists(PDF_FOLDER):
    branches = [
        file.replace(".pdf", "")
        for file in os.listdir(PDF_FOLDER)
        if file.endswith(".pdf")
    ]
else:
    branches = []

selected_branch = st.selectbox("Choose Branch", branches)

# ===============================
# ANALYSIS BUTTON
# ===============================

if st.button("🚀 Analyze Syllabus"):

    # Validate inputs
    if college_upload_method == "Upload PDF File" and not college_pdf:
        st.warning("Please upload your college syllabus.")
        st.stop()
    elif college_upload_method == "Provide Syllabus Link" and not college_url:
        st.warning("Please enter a valid College Syllabus link.")
        st.stop()

    gate_pdf_path = None
    if not selected_branch:
         st.warning("Please select a GATE branch.")
         st.stop()
    gate_pdf_path = os.path.join(PDF_FOLDER, f"{selected_branch}.pdf")
    if not os.path.exists(gate_pdf_path):
        st.error("Selected GATE syllabus not found in backend.")
        st.stop()

    with st.spinner("Preparing files and running Semantic AI Analysis..."):

        college_temp_path = None

        try:
            # Process College Text
            if college_upload_method == "Upload PDF File":
                college_text = extract_pdf_text(college_pdf)
            else:
                response = requests.get(college_url)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(response.content)
                    college_temp_path = tmp.name
                college_text = extract_pdf_text(college_temp_path)

            # Process GATE Text
            if gate_pdf_path is None:
                st.error("Error: Selected GATE syllabus path is invalid.")
                st.stop()
            with open(str(gate_pdf_path), "rb") as f:
                gate_text = extract_pdf_text(f)

        except Exception as e:
             st.error(f"Error processing PDFs: {e}")
             st.stop()
        finally:
             if college_temp_path and os.path.exists(college_temp_path):
                 os.remove(college_temp_path)

        overall_similarity = compute_overall_similarity(
            college_text, gate_text
        )

        college_topics = extract_topics(college_text)
        gate_topics = extract_topics(gate_text)

        results = topic_wise_similarity(college_topics, gate_topics)

    # ===============================
    # OUTPUT SECTION
    # ===============================

    st.subheader("📊 Overall Syllabus Similarity")
    st.success(f"{overall_similarity:.2f}%")

    st.subheader("🔎 Topic-wise Similarity & Priority")
    st.dataframe([{
        "GATE Topic": r["gate_topic"],
        "Matched College Topic": r["matched_topic"],
        "Similarity (%)": f"{r['similarity']:.2f}",
        "Priority": r["priority"]
    } for r in results])

    st.subheader("🔥 High Priority Topics - Recommended Lectures")

    for r in results:
        if r["priority"] == "High":
            st.markdown(f"### 📌 {r['gate_topic']}")
            video_link = get_best_video_link(r["gate_topic"])
            if video_link:
                st.markdown(f"[Watch Recommended Lecture]({video_link})")
            st.divider()