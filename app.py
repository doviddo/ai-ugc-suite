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

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload
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

os.makedirs('temp', exist_ok=True)
os.makedirs('output', exist_ok=True)

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


def analyze_with_gemini(file_path, product_context, mode):
    """Send file to Gemini for creative concept generation using official SDK."""
    mime_type = get_mime_type(file_path)
    is_video = mime_type.startswith('video')

    if is_video:
        prompt = f"""You are a professional UGC video marketing expert for the German market.

Analyze this video footage and the product information below.
Create a compelling 15-second German voiceover script for this UGC-style content.

PRODUCT CONTEXT / MARKETING INFO:
{product_context}

REQUIREMENTS:
- The video shows someone interacting with the product. Write a voiceover that matches what's happening visually.
- Return ONLY a valid JSON object with exactly two fields:
  1) "voiceover_script" - enthusiastic German voiceover, informal "du"-style, 15 seconds when spoken, first person, as if a male reviewer speaks to camera
  2) "video_description" - brief English description of what happens in the video (for our records)
- voiceover_script MUST be in German only.
- Make it feel natural, not robotic. Use pauses (...) for effect."""
    else:
        prompt = f"""You are a professional UGC video marketing expert for the German market.

Analyze this product photo and the marketing context below.
Create a complete UGC video concept.

PRODUCT CONTEXT / MARKETING INFO:
{product_context}

REQUIREMENTS:
- Return ONLY a valid JSON object with exactly two fields:
  1) "video_prompt" - a detailed 15-second cinematic description in English for Google Veo 3. Focus on PRODUCT SHOTS and HANDS only — show hands unboxing, holding, demonstrating the product against a clean background. NO full person, NO face. Cinematic close-ups, smooth camera movements, premium lighting. Silent, no speech.
  2) "voiceover_script" - a catchy 15-second enthusiastic script ENTIRELY IN GERMAN language, first person, informal "du"-style, as if a male reviewer is speaking directly to camera. Use the product marketing info to highlight key benefits. Make it feel authentic and exciting.
- voiceover_script MUST be in German only."""

    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    response = gemini_client.models.generate_content(
        model=ANALYSIS_MODEL,
        contents=[prompt, file_part],
        config=types.GenerateContentConfig(
            response_mime_type='application/json'
        )
    )

    return json.loads(response.text)


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
        raise RuntimeError(f"TTS response missing audio data: {e}. Response: {data}")

    return base64.b64decode(audio_b64)


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


def merge_audio_video(video_path, audio_raw_path, output_path, video_duration):
    """Merge audio and video with FFmpeg, adjusting audio speed to fit video duration."""
    wav_path = audio_raw_path.replace('.raw', '.wav')

    # Try s16le first (little-endian), fallback to s16be (big-endian)
    # Gemini TTS returns L16 PCM - format can vary
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

    # Get audio duration
    audio_duration = get_video_duration(wav_path)

    # Calculate speed ratio to fit audio into video duration
    audio_filter = []
    if audio_duration > 0 and abs(audio_duration - video_duration) > 0.5:
        speed_ratio = audio_duration / video_duration
        speed_ratio = max(0.5, min(2.0, speed_ratio))  # clamp to safe range
        audio_filter = ['-filter:a', f'atempo={speed_ratio:.4f}']

    # If video is shorter than audio, loop it; if longer, trim to audio length
    if audio_duration > 0 and video_duration < audio_duration:
        # Loop video to cover full audio duration
        loop_count = int(audio_duration / video_duration) + 2
        cmd = [
            'ffmpeg', '-y',
            '-stream_loop', str(loop_count), '-i', video_path,
            '-i', wav_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
    else:
        # Video is long enough — just merge and trim to audio length
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', wav_path,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '128k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f'FFmpeg merge error: {result.stderr[-500:]}')
    return output_path


def process_job(job_id, mode, file_path, product_context, voiceover_script=None, video_prompt=None, voice=None):
    """Background thread to process the full video generation pipeline."""
    try:
        jobs[job_id]['status'] = 'generating_audio'

        audio_raw_path = f"temp/{job_id}_audio.raw"
        audio_data = generate_tts(voiceover_script, voice)
        with open(audio_raw_path, 'wb') as f:
            f.write(audio_data)

        if mode == 'creative':
            jobs[job_id]['status'] = 'generating_video'
            video_data = generate_veo3_video(video_prompt)
            video_path = f"temp/{job_id}_video.mp4"
            with open(video_path, 'wb') as f:
                f.write(video_data)
        else:
            # mode == 'dubbing' — use uploaded video, strip original audio
            jobs[job_id]['status'] = 'processing_video'
            stripped_path = f"temp/{job_id}_video.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path,
                '-an', '-c:v', 'copy',
                stripped_path
            ], check=True, capture_output=True, timeout=120)
            video_path = stripped_path

        jobs[job_id]['status'] = 'merging'
        video_duration = get_video_duration(video_path)
        output_path = f"output/{job_id}_final.mp4"
        merge_audio_video(video_path, audio_raw_path, output_path, video_duration)

        jobs[job_id]['status'] = 'done'
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    product_context = request.form.get('product_context', '').strip()

    if not product_context:
        return jsonify({'error': 'Product context is required'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({'error': 'Invalid filename'}), 400

    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    if ext in ALLOWED_PHOTO_EXT:
        mode = 'creative'
    elif ext in ALLOWED_VIDEO_EXT:
        mode = 'dubbing'
    else:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    job_id = str(uuid.uuid4())
    save_path = f"temp/{job_id}_{filename}"
    file.save(save_path)

    try:
        creative_data = analyze_with_gemini(save_path, product_context, mode)
    except Exception as e:
        os.remove(save_path)
        return jsonify({'error': f'Gemini analysis failed: {str(e)}'}), 500

    jobs[job_id] = {
        'status': 'pending_confirmation',
        'mode': mode,
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

    return jsonify(response)


@app.route('/download/<filename>')
def download(filename):
    """Download the final video."""
    return send_from_directory('output', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
