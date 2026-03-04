import os
import uuid
import json
import time
import base64
import subprocess
import threading
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai
from google.genai import types
import yt_dlp

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2GB max upload
app.config['UPLOAD_FOLDER'] = 'temp'

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
VEO_API_KEY = os.getenv('VEO_API_KEY')
TTS_API_KEY = os.getenv('TTS_API_KEY')
TTS_VOICE = os.getenv('TTS_VOICE', 'Puck')

ANALYSIS_MODEL = 'gemini-2.5-flash'
TTS_MODEL = 'gemini-2.5-flash-preview-tts'

# Initialize google-genai client for analysis
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

ALLOWED_PHOTO_EXT = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXT = {'mp4', 'mov', 'avi', 'webm'}

os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("library", exist_ok=True)
os.makedirs('assets', exist_ok=True)
os.makedirs('assets/sfx', exist_ok=True)

BG_MUSIC_PATH = 'assets/bg_music.mp3'
if not os.path.exists(BG_MUSIC_PATH):
    print("Downloading default background music...")
    try:
        # A reliable public domain test track, can be replaced by user later
        resp = requests.get("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", timeout=60)
        with open(BG_MUSIC_PATH, 'wb') as f:
            f.write(resp.content)
    except Exception as e:
        print(f"Failed to download bg music: {e}")

# SFX library: key -> (emoji label, filename)
SFX_LIBRARY = {
    'unboxing': ('📦 Unboxing', 'assets/sfx/unboxing.mp3'),
    'drone':    ('🚁 Drone flying', 'assets/sfx/drone.mp3'),
    'click':    ('🖱️ Click / Snap', 'assets/sfx/click.mp3'),
    'whoosh':   ('💨 Whoosh', 'assets/sfx/whoosh.mp3'),
    'crowd':    ('👏 Crowd / Applause', 'assets/sfx/crowd.mp3'),
    'kitchen':  ('🍳 Kitchen sounds', 'assets/sfx/kitchen.mp3'),
}

# Generate silent placeholder SFX if real files don't exist
for _sfx_key, (_sfx_label, _sfx_path) in SFX_LIBRARY.items():
    if not os.path.exists(_sfx_path):
        try:
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                '-t', '10', _sfx_path
            ], capture_output=True, timeout=15)
        except Exception:
            pass

# In-memory job store
jobs = {}


def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def get_video_duration(filepath):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'json', filepath
        ], capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception:
        return 15.0  # Default fallback


def file_to_base64(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_mime_type(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    mime_map = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
        'png': 'image/png', 'webp': 'image/webp',
        'mp4': 'video/mp4', 'mov': 'video/quicktime',
        'avi': 'video/avi', 'webm': 'video/webm'
    }
    return mime_map.get(ext, 'application/octet-stream')


def analyze_with_gemini(file_path, product_context, mode, video_duration_sec=None, clip_count=10):
    """
    Sends the file (image or video) and marketing context to Gemini 1.5 Flash.
    Forces JSON output with response_schema for reliability.
    """
    prompt = ""
    is_video = bool(video_duration_sec)
    mime_type = "video/mp4" if is_video else "image/jpeg"

    clip_count = max(1, min(10, int(clip_count))) # clamp between 1-10

    if mode == 'dubbing':
        duration_hint = f"{int(video_duration_sec)} seconds" if video_duration_sec else "15 seconds"
        prompt = f"""You are a professional UGC video marketing expert for the Russian market.

Analyze this video footage and the product information below.
Create a Russian voiceover script that fits EXACTLY {duration_hint} of video.

PRODUCT CONTEXT / MARKETING INFO:
{product_context}

REQUIREMENTS:
- The video shows someone interacting with the product. Write a voiceover that matches what's happening visually.
- The script MUST be readable in exactly {duration_hint} at a natural speaking pace.
- Return ONLY a valid JSON object with exactly two fields:
  1) "voiceover_script" - enthusiastic Russian voiceover, informal "ты"-style, first person, as if a male reviewer speaks to camera. Length: {duration_hint} when spoken at normal pace.
  2) "video_description" - brief English description of what happens in the video (for our records)
- voiceover_script MUST be in Russian only.
- Make it feel natural, not robotic. Use pauses (...) for effect."""
    
    elif mode == 'clipper':
        clip_count = 1 # Force 1 clip based on user prompt directly
        prompt = f"""You are a top-tier TikTok creator and storyteller specializing in psychological thriller movie recaps for the Russian market.

🧠 PSYCHOLOGICAL THRILLER MODE ACTIVATED.
Analyze this video footage and the context below.
Your task is to create a viral, engaging 20-30 second TikTok video analyzing or retelling the footage as a deep psychological thriller.

USER PROMPT / CONTEXT:
{product_context}

REQUIREMENTS FOR THIS VIDEO:
1. VISUALS: Select 4 to 6 minimal fragments or stop-frames that show tension, strange expressions, or key details. Total combined duration of segments should be 20-30 seconds.
2. NARRATION: Write a brilliant Russian voiceover script that finds a deep, hidden meaning or tells the story as a intense thriller.
   - DO NOT USE boring openings like "В этом фильме парень встретил девушку…".
   - INSTEAD, USE a powerful thriller hook like "Он даже не знал, что через 48 часов его жизнь закончится…" or "Посмотрите на её глаза, вы замечаете эту жуткую деталь?".
   - Change the delivery to be mysterious, analytical, and insightful.
   - It must perfectly fit the 20-30s length when spoken.
3. Split the voiceover script into short phrases (3-6 words each) suitable for large on-screen subtitles.
4. Return ONLY a valid JSON object matching exactly this structure.
CRITICAL JSON RULES:
- ABSOLUTELY NO internal quotation marks! You are STRICTLY FORBIDDEN from using any double quotes (") or single quotes (') inside the actual text values.
- Do not use newlines in text. 
- The output MUST be strictly valid JSON without any markdown formatting.

{{
  "clips": [
    {{
      "theme": "Brief description of the angle based on prompt",
      "cut_segments": [
        {{"start_time": "1.5", "end_time": "4.0"}},
        {{"start_time": "12.0", "end_time": "15.5"}}
      ],
      "voiceover_script": "full russian script without quotes",
      "subtitles": ["phrase 1", "phrase 2"]
    }}
  ]
}}"""

    else:
        # creative
        prompt = f"""You are a professional UGC video marketing expert for the German market.

Analyze this product image and the marketing info below.
Create a German voiceover script and a prompt for a generative AI video model (like Google Veo 3) based on the image.

PRODUCT CONTEXT:
{product_context}

REQUIREMENTS:
1. Return ONLY a valid JSON object with EXACTLY three fields.
  1) "video_prompt" - a detailed cinematic description in English for Google Veo 3. Focus on PRODUCT SHOTS and HANDS only! Show hands unboxing, consuming, or using the product. NO talking heads, NO people speaking. Just aesthetic ASMR-style interactions and background actions.
  2) "voiceover_script" - an enthusiastic, powerful HOOK ENTIRELY IN GERMAN. It must catch the buyer's attention INSTANTLY! Length: strictly 5 to 6 seconds when spoken at normal pace (so it finishes smoothly before the 8-second video ends and never cuts off).
  3) "sfx_list" - array of 1-3 background sound effect names that BEST match the visual scene. Choose ONLY from this list: unboxing, drone, click, whoosh, crowd, kitchen. Use "none" if no sound fits.
- voiceover_script MUST be extremely short, punchy, and informal ("du"-style). Max 6 seconds!"""

    # Define strict JSON schemas based on mode to guarantee zero parsing errors
    if mode == 'clipper':
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "clips": types.Schema(
                    type=types.Type.ARRAY,
                    description=f"List of {clip_count} distinct video clip concepts",
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "theme": types.Schema(type=types.Type.STRING),
                            "cut_segments": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "start_time": types.Schema(type=types.Type.STRING, description="Start time (e.g., '12.5' or '01:15')"),
                                        "end_time": types.Schema(type=types.Type.STRING, description="End time (e.g., '15.0' or '01:20')")
                                    }
                                )
                            ),
                            "voiceover_script": types.Schema(type=types.Type.STRING, description="Russian voiceover script"),
                            "subtitles": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(type=types.Type.STRING, description="Short phrases for subtitles")
                            )
                        }
                    )
                )
            }
        )
    else:
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "video_prompt": types.Schema(type=types.Type.STRING, description="Prompt for Google Veo 3"),
                "voiceover_script": types.Schema(type=types.Type.STRING, description="German voiceover script"),
                "sfx_list": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING,
                        description="Sound effects from: unboxing, drone, click, whoosh, crowd, kitchen, none")
                )
            }
        )

    if is_video:
        # For large videos, we MUST use the File API to upload rather than inline memory payload
        uploaded_file = gemini_client.files.upload(
            file=file_path, 
            config=types.UploadFileConfig(mime_type=mime_type, display_name=os.path.basename(file_path)[:40])
        )
        
        # Wait for the file to be processed by Gemini backend
        while True:
            state = str(uploaded_file.state)
            if "PROCESSING" in state:
                time.sleep(5)
                uploaded_file = gemini_client.files.get(name=uploaded_file.name)
            else:
                break
                
        if "FAILED" in str(uploaded_file.state):
            raise RuntimeError("Gemini failed to process the video on their end.")
            
        media_content = uploaded_file
    else:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        media_content = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    response = gemini_client.models.generate_content(
        model=ANALYSIS_MODEL,
        contents=[prompt, media_content],
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=schema,
            temperature=0.7
        )
    )

    # Cleanup the file from Google servers to save quota
    if is_video:
        try:
            gemini_client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

    try:
        # Strip potential markdown blocks Gemini sometimes adds despite settings
        raw_text = response.text.strip()
        if raw_text.startswith('```json'):
            raw_text = raw_text[7:]
        elif raw_text.startswith('```'):
            raw_text = raw_text[3:]
        if raw_text.endswith('```'):
            raw_text = raw_text[:-3]

        # Try to fix common unescaped quote issues before parsing
        import re
        # This replaces quotes inside string values but it's risky, so we rely on strict=False
        # which permits unescaped newlines and control characters at least.
            
        return json.loads(raw_text.strip(), strict=False)
        
    except json.JSONDecodeError as e:
        print(f"Gemini raw failed response: {response.text}")
        raise RuntimeError(f"Gemini returned invalid JSON (often caused by unescaped quotes). Please try again. Error: {str(e)}")


def generate_tts(voiceover_script, voice_name=None):
    """Generate TTS audio from German script via HTTP (TTS requires specific model endpoint)."""
    voice = voice_name or TTS_VOICE
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent?key={TTS_API_KEY}"

    body = {
        "contents": [{"parts": [{"text": voiceover_script}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            }
        }
    }

    response = requests.post(url, json=body, timeout=120)
    response.raise_for_status()

    data = response.json()
    try:
        audio_b64 = data['candidates'][0]['content']['parts'][0]['inlineData']['data']
    except (KeyError, IndexError, TypeError) as e:
        safe_text = voiceover_script[:50] if voiceover_script else "EMPTY"
        raise RuntimeError(f"TTS response missing audio data for text '{safe_text}...'. Error: {e}. Response: {data}")

    return base64.b64decode(audio_b64)


def generate_srt(subtitles_list, audio_duration, filename):
    """Generate proportional .srt file from a list of subtitle phrases."""
    total_chars = sum(len(text.strip()) for text in subtitles_list)
    if total_chars == 0:
        return
    
    current_time = 0.0
    with open(filename, 'w', encoding='utf-8') as f:
        for i, text in enumerate(subtitles_list, 1):
            text = text.strip()
            if not text:
                continue
            char_count = len(text)
            duration = (char_count / total_chars) * audio_duration
            
            start_time = current_time
            end_time = current_time + duration
            
            # Format to SRT timestamp HH:MM:SS,MMM
            def format_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            f.write(f"{i}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"{text}\n\n")
            
            current_time = end_time


def generate_veo3_video(video_prompt):
    """Submit video generation to Veo 3 and poll until complete."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-001:predictLongRunning?key={VEO_API_KEY}"

    body = {"instances": [{"prompt": video_prompt}]}
    response = requests.post(url, json=body, timeout=60)
    response.raise_for_status()
    operation_name = response.json()['name']

    # Poll for completion
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={VEO_API_KEY}"
    for _ in range(30):  # max 5 minutes (10s * 30)
        time.sleep(10)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_data = poll_resp.json()
        if poll_data.get('done'):
            # Check for API-level error
            if 'error' in poll_data:
                raise RuntimeError(f"Veo 3 API error: {poll_data['error']}")

            veo_response = poll_data.get('response', {}).get('generateVideoResponse', {})

            # Content safety filter rejection
            if 'raiFilteredReason' in veo_response:
                reason = veo_response['raiFilteredReason']
                raise RuntimeError(
                    f"Veo 3 rejected the prompt due to content safety filter: {reason}. "
                    f"Try simplifying the video description (avoid 'person', 'reviewer', 'human')."
                )

            # Try generatedSamples (v1) or generatedVideos (newer API versions)
            samples = veo_response.get('generatedSamples') or veo_response.get('generatedVideos')
            if not samples:
                raise RuntimeError(f"Veo 3 returned no video samples. Full response: {poll_data}")

            video_uri = samples[0].get('video', {}).get('uri') or samples[0].get('uri')
            if not video_uri:
                raise RuntimeError(f"Veo 3 video URI not found in response: {samples}")

            # Download video
            video_url = f"{video_uri}&key={VEO_API_KEY}"
            video_resp = requests.get(video_url, timeout=120)
            return video_resp.content
    raise TimeoutError("Veo 3 video generation timed out after 5 minutes")


def merge_audio_video(video_path, audio_raw_path, output_path, video_duration, aspect_ratio='vertical', sfx_list=None):
    """Merge audio and video with FFmpeg, apply scale/crop and optionally mix SFX."""
    wav_path = audio_raw_path.replace('.raw', '.wav')

    # Determine target resolution
    if aspect_ratio == 'horizontal':
        scale_crop = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"
    elif aspect_ratio == 'fb_feed':
        scale_crop = "scale=1080:1350:force_original_aspect_ratio=increase,crop=1080:1350"
    else:  # vertical (default)
        scale_crop = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

    # Try s16le first (little-endian), fallback to s16be (big-endian)
    converted = False
    for fmt in ['s16le', 's16be']:
        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', fmt, '-ar', '24000', '-ac', '1',
                '-i', audio_raw_path,
                wav_path
            ], capture_output=True, timeout=60)
            if result.returncode == 0:
                converted = True
                break
        except Exception:
            continue

    if not converted:
        raise RuntimeError('Failed to convert TTS audio to WAV')

    # Determine which SFX files to mix
    valid_sfx_paths = []
    if sfx_list:
        for sfx_key in sfx_list:
            if sfx_key in SFX_LIBRARY:
                sfx_path = SFX_LIBRARY[sfx_key][1]
                if os.path.exists(sfx_path):
                    valid_sfx_paths.append(sfx_path)

    # Build FFmpeg command with optional SFX mixing
    cmd = ['ffmpeg', '-y', '-i', video_path, '-i', wav_path]
    for sfx_path in valid_sfx_paths:
        cmd += ['-stream_loop', '-1', '-i', sfx_path]

    # Build audio filter: voice at 1.0, each SFX at 0.25
    num_inputs = 2 + len(valid_sfx_paths)
    if valid_sfx_paths:
        audio_parts = ['[1:a]volume=1.0[voice]']
        mix_inputs = '[voice]'
        for idx, _ in enumerate(valid_sfx_paths):
            audio_parts.append(f'[{idx+2}:a]volume=0.25[sfx{idx}]')
            mix_inputs += f'[sfx{idx}]'
        audio_parts.append(f'{mix_inputs}amix=inputs={1+len(valid_sfx_paths)}:duration=first:dropout_transition=0[aout]')
        audio_filter = ';'.join(audio_parts)
        filter_complex = f'[0:v]{scale_crop},tpad=stop_mode=clone:stop_duration=5[vout];{audio_filter}'
        cmd += ['-filter_complex', filter_complex, '-map', '[vout]', '-map', '[aout]']
    else:
        filter_complex = f'[0:v]{scale_crop},tpad=stop_mode=clone:stop_duration=5[vout]'
        cmd += ['-filter_complex', filter_complex, '-map', '[vout]', '-map', '1:a:0']

    cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k', '-shortest', output_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg merge error: {result.stderr[-500:]}')
    return output_path

def parse_time_to_sec(val):
    """Convert string formats like '1:24', '01:24.5', or '124' to float seconds."""
    if not val:
        return 0.0
    val_str = str(val).strip()
    if ':' in val_str:
        parts = val_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    try:
        return float(val_str)
    except Exception:
        return 0.0


def process_job(job_id, mode, file_path, product_context, voiceover_script=None, video_prompt=None, voice=None):
    """Background thread to process the full video generation pipeline."""
    try:
        if mode != 'clipper':
            jobs[job_id]['status'] = 'generating_audio'
            # 1. Generate Voice
            audio_raw_path = f"temp/{job_id}_audio.raw"
            audio_data = generate_tts(voiceover_script, voice)
            with open(audio_raw_path, 'wb') as f:
                f.write(audio_data)

            # 2. Convert Voice to WAV right away to get duration
            wav_path = audio_raw_path.replace('.raw', '.wav')
            
            converted = False
            for fmt in ['s16le', 's16be']:
                try:
                    res = subprocess.run([
                        'ffmpeg', '-y', '-f', fmt, '-ar', '24000', '-ac', '1',
                        '-i', audio_raw_path, wav_path
                    ], capture_output=True, timeout=60)
                    if res.returncode == 0:
                        converted = True
                        break
                except Exception:
                    continue
            if not converted:
                raise RuntimeError('Failed to convert TTS audio to WAV')
                
            audio_duration = get_video_duration(wav_path)

        # ── Mode-specific video pipeline ──
        if mode == 'creative':
            jobs[job_id]['status'] = 'generating_video'
            video_data = generate_veo3_video(video_prompt)
            video_path = f"temp/{job_id}_video.mp4"
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            jobs[job_id]['status'] = 'merging'
            output_path = f"output/{job_id}_final.mp4"
            aspect_ratio = jobs[job_id].get('aspect_ratio', 'vertical')
            sfx_list = jobs[job_id].get('sfx_list', [])
            merge_audio_video(video_path, audio_raw_path, output_path, get_video_duration(video_path), aspect_ratio, sfx_list)

        elif mode == 'dubbing':
            jobs[job_id]['status'] = 'processing_video'
            video_path = f"temp/{job_id}_video.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path, '-an', '-c:v', 'copy', video_path
            ], check=True, capture_output=True, timeout=120)

            jobs[job_id]['status'] = 'merging'
            output_path = f"output/{job_id}_final.mp4"
            aspect_ratio = jobs[job_id].get('aspect_ratio', 'vertical')
            sfx_list = jobs[job_id].get('sfx_list', [])
            merge_audio_video(video_path, audio_raw_path, output_path, get_video_duration(video_path), aspect_ratio, sfx_list)
            
        elif mode == 'clipper':
            creative_data = jobs[job_id]['creative_data']
            clips = creative_data.get('clips', [])
            if not clips:
                # Fallback if AI didn't return 'clips' array properly
                clips = [creative_data]
            
            jobs[job_id]['output_files'] = []
            
            for clip_idx, clip_data in enumerate(clips):
                jobs[job_id]['status'] = f'processing_clip_{clip_idx + 1}_of_{len(clips)}'
                
                # A. Generate Audio for this specific clip
                clip_voice_script = clip_data.get('voiceover_script')
                # If Gemini forgot the script, or user entered something too short, provide a valid fallback
                if not clip_voice_script or len(str(clip_voice_script).strip()) < 3:
                    clip_voice_script = "Hier ist ein automatisch generiertes Video für dieses Produkt."
                    
                audio_raw_path = f"temp/{job_id}_clip{clip_idx}_audio.raw"
                audio_data = generate_tts(clip_voice_script, voice)
                with open(audio_raw_path, 'wb') as f:
                    f.write(audio_data)
                
                # Convert to WAV
                wav_path = audio_raw_path.replace('.raw', '.wav')
                for fmt in ['s16le', 's16be']:
                    try:
                        res = subprocess.run([
                            'ffmpeg', '-y', '-f', fmt, '-ar', '24000', '-ac', '1',
                            '-i', audio_raw_path, wav_path
                        ], capture_output=True, timeout=60)
                        if res.returncode == 0: break
                    except Exception: continue
                
                audio_duration = get_video_duration(wav_path)

                # B. Generate Subtitles
                subtitles_list = clip_data.get('subtitles', [])
                srt_path = f"temp/{job_id}_clip{clip_idx}.srt"
                generate_srt(subtitles_list, audio_duration, srt_path)
                abs_srt_path = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
                
                # C. Prepare FFmpeg Filter Complex for Segments
                source_duration = get_video_duration(file_path)
                segments = clip_data.get('cut_segments', [{'start_time': 0, 'end_time': 20}])
                filter_parts = []
                concat_inputs = []
                valid_segments_count = 0
                
                for i, seg in enumerate(segments):
                    start = parse_time_to_sec(seg.get('start_time', 0))
                    end = parse_time_to_sec(seg.get('end_time', 5))
                    
                    # Clamp to actual video duration to prevent ffmpeg crashing or empty video
                    start = max(0.0, min(start, source_duration - 0.5))
                    end = max(start + 0.5, min(end, source_duration))
                    
                    if start >= source_duration - 0.2:
                        continue # Skip this broken segment

                    filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{valid_segments_count}]")
                    concat_inputs.append(f"[v{valid_segments_count}]")
                    valid_segments_count += 1
                    
                if valid_segments_count == 0:
                    fallback_end = min(5.0, source_duration)
                    filter_parts.append(f"[0:v]trim=start=0:end={fallback_end},setpts=PTS-STARTPTS[v0]")
                    concat_inputs.append("[v0]")
                    valid_segments_count = 1
                    
                concat_str = "".join(concat_inputs)
                filter_parts.append(f"{concat_str}concat=n={valid_segments_count}:v=1:a=0[vcat]")
                
                # Apply scaling/cropping based on aspect ratio
                aspect_ratio = jobs[job_id].get('aspect_ratio', 'vertical')
                if aspect_ratio == 'horizontal':
                    scale_filter = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"
                elif aspect_ratio == 'fb_feed':
                    scale_filter = "scale=1080:1350:force_original_aspect_ratio=increase,crop=1080:1350"
                else:
                    scale_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
                
                # Add tpad to freeze the last frame for up to 30s so the video doesn't end before the long German voiceover
                filter_parts.append(f"[vcat]{scale_filter},tpad=stop_mode=clone:stop_duration=30[vscaled]")
                
                # Apply subtitles on the scaled video
                # White color (&H00FFFFFF), smaller font (FontSize=16), closer to bottom (MarginV=15)
                filter_parts.append(f"[vscaled]subtitles={abs_srt_path}:force_style='FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H40000000,BorderStyle=1,Outline=1,Shadow=1,MarginV=15,Bold=1'[vout]")
                
                filter_complex = ";".join(filter_parts)
                
                # Audio mixing: Voice (1.0) + Looped BG Music (0.1)
                audio_filter = "[1:a]volume=1.0[v_aud];[2:a]volume=0.1[bg_aud];[v_aud][bg_aud]amix=inputs=2:duration=first:dropout_transition=0[aout]"
                filter_complex += f";{audio_filter}"
                
                output_path = f"output/{job_id}_clip{clip_idx + 1}.mp4"
                
                bg_music_args = []
                if os.path.exists(BG_MUSIC_PATH):
                    bg_music_args = ['-stream_loop', '-1', '-i', BG_MUSIC_PATH]
                else:
                    # Fallback if bg music didn't download
                    bg_music_args = ['-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono']
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', file_path,
                    '-i', wav_path
                ] + bg_music_args + [
                    '-filter_complex', filter_complex,
                    '-map', '[vout]',
                    '-map', '[aout]',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-shortest',
                    output_path
                ]
                
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                if res.returncode != 0:
                    print(f"Clip {clip_idx} FFmpeg error: {res.stderr[-500:]}")
                    continue # Try to generate the next clips even if one fails
                
                jobs[job_id]['output_files'].append(f"{job_id}_clip{clip_idx + 1}.mp4")

        jobs[job_id]['status'] = 'done'
        if mode != 'clipper':
            jobs[job_id]['output_file'] = f"{job_id}_final.mp4"


    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
    finally:
        # Cleanup temp files
        for f in [file_path, audio_raw_path, f"temp/{job_id}_video.mp4"]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Step 1: Upload file + context, get Gemini creative concept."""
    product_context = request.form.get('product_context', '').strip()
    mode = request.form.get('mode', '').strip()  # Mode must come from frontend!
    aspect_ratio = request.form.get('aspect_ratio', 'vertical').strip()
    video_url = request.form.get('video_url', '').strip()
    
    try:
        clip_count = int(request.form.get('clip_count', 10))
    except (ValueError, TypeError):
        clip_count = 10

    selected_video = request.form.get('selected_video', '').strip()
    file = request.files.get('file')

    if not file and not video_url and not selected_video:
        return jsonify({'error': 'Please provide either a file, video URL, or select from library'}), 400

    job_id = str(uuid.uuid4())
    
    if selected_video:
        save_path = f"library/{secure_filename(selected_video)}"
        if not os.path.exists(save_path):
            return jsonify({'error': 'Selected video not found in library'}), 404
    elif video_url:
        save_path = f"library/{job_id}_downloaded.mp4"
        ydl_opts = {
            'outtmpl': save_path,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'noplaylist': True,
            'extractor_args': {'youtube': ['player_client=android', 'player_skip=web']}
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            return jsonify({'error': f'Failed to download video from URL: {str(e)}'}), 400
    else:
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        save_path = f"library/{filename}"
        file.save(save_path)

    # For video modes — measure duration
    video_duration_sec = None
    if mode in ['dubbing', 'clipper']:
        video_duration_sec = get_video_duration(save_path)

    try:
        creative_data = analyze_with_gemini(save_path, product_context, mode, video_duration_sec, clip_count)
    except Exception as e:
        if not selected_video:
            try: os.remove(save_path)
            except: pass
            
        err = str(e)
        if "Unable to process input" in err:
            return jsonify({'error': 'Gemini AI failed to read this video format. It might use an unsupported codec. Please try converting it to a standard MP4 or use a different video.'}), 400
            
        return jsonify({'error': f'Gemini analysis failed: {err}'}), 500

    jobs[job_id] = {
        'status': 'pending_confirmation',
        'mode': mode,
        'aspect_ratio': aspect_ratio,
        'file_path': save_path,
        'creative_data': creative_data,
        'product_context': product_context
    }

    return jsonify({
        'job_id': job_id,
        'mode': mode,
        'creative_data': creative_data
    })


@app.route('/generate', methods=['POST'])
def generate():
    """Step 2: User confirms/edits script, start full generation."""
    data = request.json
    job_id = data.get('job_id')
    voiceover_script = data.get('voiceover_script', '').strip()
    voice = data.get('voice', TTS_VOICE)
    sfx_list = data.get('sfx_list', [])

    if not job_id or job_id not in jobs:
        return jsonify({'error': 'Invalid job ID'}), 400

    if not voiceover_script:
        return jsonify({'error': 'Voiceover script cannot be empty'}), 400

    job = jobs[job_id]
    if job['status'] != 'pending_confirmation':
        return jsonify({'error': 'Job is not in confirmation state'}), 400

    mode = job['mode']
    creative_data = job['creative_data']
    video_prompt = creative_data.get('video_prompt', '')
    file_path = job['file_path']

    jobs[job_id]['status'] = 'starting'
    jobs[job_id]['sfx_list'] = [s for s in sfx_list if s and s != 'none']

    thread = threading.Thread(
        target=process_job,
        args=(job_id, mode, file_path, job['product_context'], voiceover_script, video_prompt, voice),
        daemon=True
    )
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'started'})


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    """Poll job status."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    response = {'status': job['status']}

    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    elif job['status'] == 'done':
        response['output_file'] = job.get('output_file')
        response['output_files'] = job.get('output_files')
        response['creative_data'] = job.get('creative_data')

    return jsonify(response)


@app.route('/download/<filename>')
def download(filename):
    """Download the final video."""
    return send_from_directory('output', filename, as_attachment=True)

@app.route('/api/library', methods=['GET'])
def list_library():
    """List videos available in the library folder."""
    if not os.path.exists("library"): return jsonify([])
    files = [f for f in os.listdir("library") if f.endswith(('.mp4', '.mov', '.avi', '.webm'))]
    return jsonify(files)

@app.route('/api/upload_to_library', methods=['POST'])
def upload_to_library():
    """Upload a video directly to the persistent library."""
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'}), 400
    filename = secure_filename(file.filename)
    if not filename: return jsonify({'error': 'Invalid filename'}), 400
    save_path = f"library/{filename}"
    file.save(save_path)
    return jsonify({'success': True, 'filename': filename})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
