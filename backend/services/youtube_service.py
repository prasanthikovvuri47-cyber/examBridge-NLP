import os
import re
from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_client():
    if not YOUTUBE_API_KEY:
        return None
    try:
        return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        print(f"YouTube API Error: {e}")
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
    youtube = get_youtube_client()
    if youtube is None:
        return [{"error": "YouTube API key not configured or invalid"}]

    try:
        # Search for long-form lectures
        search = youtube.search().list(
            q=f"{query} technical lecture",
            part="snippet", type="video", videoDuration="long",
            order="viewCount", maxResults=max_results * 2
        ).execute()

        video_ids = [item['id']['videoId'] for item in search.get('items', [])]
        
        if not video_ids:
            # Fallback to medium duration if no long ones found
            search = youtube.search().list(
                q=query, part="snippet", type="video", videoDuration="medium",
                order="viewCount", maxResults=max_results
            ).execute()
            video_ids = [item['id']['videoId'] for item in search.get('items', [])]

        if not video_ids:
            return []

        # Get statistics for scoring
        videos_data = youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=",".join(video_ids)
        ).execute()

        scored_videos = []
        for item in videos_data.get('items', []):
            duration_sec = parse_duration(item['contentDetails'].get('duration', ''))
            if duration_sec < 300: continue # Skip shorts/very short videos

            views = int(item['statistics'].get('viewCount', 0))
            likes = int(item['statistics'].get('likeCount', 0))
            score = views + (likes * 50)

            scored_videos.append({
                "title": item['snippet']['title'],
                "channel": item['snippet']['channelTitle'],
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "thumbnail": item['snippet']['thumbnails']['high']['url'],
                "duration": item['contentDetails']['duration'].replace('PT', '').lower(),
                "score": score
            })

        # Sort by score and return top results
        scored_videos.sort(key=lambda x: x['score'], reverse=True)
        return scored_videos[:max_results]

    except Exception as e:
        print(f"YouTube Fetch Error: {e}")
        return [{"error": str(e)}]