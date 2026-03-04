import os
import re
import logging
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    # Read key dynamically every time to support load_dotenv() and env var changes
    api_key = os.getenv("YOUTUBE_API_KEY") or YOUTUBE_API_KEY
    if not api_key:
        logger.error("YOUTUBE_API_KEY is not set in environment variables.")
        return None
    try:
        client = build("youtube", "v3", developerKey=api_key)
        logger.info("YouTube API client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize YouTube API client: {e}")
        return None

def parse_duration(duration_str):
    if not duration_str: return 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match: return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def fetch_youtube_videos(query: str, max_results: int = 5):
    logger.info(f"Fetching YouTube videos for query: {query}")
    youtube = get_youtube_client()
    if youtube is None:
        logger.warning("YouTube client not available. Returning error information.")
        return [{"error": "YouTube API key not configured or invalid"}]

    try:
        # Search for long-form lectures
        logger.info(f"Searching for long-form lectures for: {query}")
        search = youtube.search().list(
            q=f"{query} technical lecture",
            part="snippet", type="video", videoDuration="long",
            order="viewCount", maxResults=max_results * 2
        ).execute()

        video_ids = [item['id']['videoId'] for item in search.get('items', [])]
        
        if not video_ids:
            # Fallback to medium duration if no long ones found
            logger.info(f"No long-form videos found for '{query}', falling back to medium duration.")
            search = youtube.search().list(
                q=query, part="snippet", type="video", videoDuration="medium",
                order="viewCount", maxResults=max_results
            ).execute()
            video_ids = [item['id']['videoId'] for item in search.get('items', [])]

        if not video_ids:
            logger.info(f"No videos found for topic: {query}")
            return []

        # Get statistics for scoring
        logger.info(f"Fetching details for {len(video_ids)} videos...")
        videos_data = youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=",".join(video_ids)
        ).execute()

        scored_videos = []
        TRUSTED_CHANNELS = [
            "gate smashers", "neso academy", "knowledge gate",
            "unacademy gate", "simply learn", "ravindrababu ravula",
            "last moment tuitions", "geeks for geeks"
        ]
        for item in videos_data.get('items', []):
            duration_sec = parse_duration(item['contentDetails'].get('duration', ''))
            if duration_sec < 300: continue # Skip shorts/very short videos

            views = int(item['statistics'].get('viewCount', 0))
            likes = int(item['statistics'].get('likeCount', 0))
            channel_name = item['snippet']['channelTitle']
            
            # Primary score: views + weighted likes
            score = views + (likes * 50)

            # Apply 1.5x boost for trusted educational channels
            if any(tc in channel_name.lower() for tc in TRUSTED_CHANNELS):
                score = int(score * 1.5)
                logger.info(f"Trusted channel boost applied to: {channel_name}")

            scored_videos.append({
                "title": item['snippet']['title'],
                "channel": channel_name,
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "thumbnail": item['snippet']['thumbnails']['high']['url'],
                "duration": item['contentDetails']['duration'].replace('PT', '').lower(),
                "score": score,
                "views": views,
                "likes": likes
            })

        # Sort by score and return top results
        scored_videos.sort(key=lambda x: x['score'], reverse=True)
        final_results = []
        for i in range(min(len(scored_videos), max_results)):
            final_results.append(scored_videos[i])
            
        logger.info(f"Successfully fetched and scored {len(final_results)} videos.")
        return final_results

    except Exception as e:
        logger.error(f"YouTube Fetch Error: {e}")
        return [{"error": str(e)}]

def get_video_summary(video_id, topic, model, util):
    """
    Generates a memory-efficient extractive summary by ranking transcript sentences 
    relative to the target topic.
    """
    logger.info(f"Generating summary for video ID: {video_id} (Topic: {topic})")
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t['text'] for t in transcript_list])
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', full_text) if len(s.strip()) > 20]
        
        if not sentences:
            logger.warning(f"No substantial content found in transcript for video {video_id}")
            return "Transcript available but no substantial content found for summary."

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
        logger.info(f"Summary generated successfully for video {video_id}")
        return summary
    except Exception as e:
        logger.warning(f"Summary generation failed for video {video_id}: {e}")
        return "Summary not available (transcripts might be disabled for this video)."
