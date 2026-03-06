import os, requests, time, json, subprocess
from dotenv import load_dotenv

load_dotenv()
VEO_API_KEY = os.getenv('VEO_API_KEY')

def get_video_duration(filepath):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'json', filepath
        ], capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception:
        return 8.0

def apply_outro_and_cover(input_path, output_path, cover_path):
    duration = get_video_duration(input_path)
    subprocess.run([
        'ffmpeg', '-y', '-ss', str(min(1.0, duration/2)), '-i', input_path,
        '-vf', "drawtext=text='www.techflug.de':fontcolor=white:fontsize=50:borderw=3:bordercolor=black:x=(w-text_w)/2:y=60",
        '-vframes', '1', '-q:v', '5', cover_path
    ], capture_output=True)
    
    filter_complex = f"[0:v]tpad=stop_mode=clone:stop_duration=2,drawtext=text='www.techflug.de':fontcolor=white:fontsize=65:borderw=3:bordercolor=black:x=(w-text_w)/2:y=(h-text_h)/2:enable='gt(t,{duration})'[vout];[0:a]apad=pad_dur=2[aout]"
    
    res = subprocess.run([
        'ffmpeg', '-y', '-i', input_path, '-filter_complex', filter_complex,
        '-map', '[vout]', '-map', '[aout]',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', output_path
    ], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FFmpeg Error: {res.stderr}")

def generate_veo3_video(video_prompt, aspect_ratio='vertical'):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-3.0-generate-001:predictLongRunning?key={VEO_API_KEY}"
    veo_aspect = {'vertical': '9:16', 'horizontal': '16:9', 'fb_feed': '4:5'}.get(aspect_ratio, '9:16')
    body = {"instances": [{"prompt": video_prompt}], "parameters": {"aspectRatio": veo_aspect}}
    response = requests.post(url, json=body, timeout=60)
    response.raise_for_status()
    operation_name = response.json()['name']
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={VEO_API_KEY}"
    print(f"Job started: {operation_name}. Polling...")
    for _ in range(60): 
        time.sleep(10)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_data = poll_resp.json()
        if poll_data.get('done'):
            if 'error' in poll_data: raise RuntimeError(f"Veo 3 Error: {poll_data['error']}")
            veo_resp = poll_data.get('response', {}).get('generateVideoResponse', {})
            samples = veo_resp.get('generatedSamples') or veo_resp.get('generatedVideos')
            if not samples: raise RuntimeError(f"No samples: {poll_data}")
            video_uri = samples[0].get('video', {}).get('uri') or samples[0].get('uri')
            video_resp = requests.get(f"{video_uri}&key={VEO_API_KEY}", timeout=120)
            return video_resp.content
    raise TimeoutError("Timeout")

prompt = "UGC style video, young professional man standing outside a busy coffee shop in Germany, looking frustrated at the long line. He checks the price on the menu and reacts surprised. Urban morning atmosphere, realistic, handheld smartphone footage. Casual natural behavior, authentic gestures, slightly imperfect movement. Shot on smartphone camera, handheld footage, natural lighting. Vertical video 9:16, TikTok style social media video."

try:
    print("Generating video with Veo 3...")
    content = generate_veo3_video(prompt)
    temp_p = "output/temp_raw.mp4"
    final_p = "output/coffee_shop_ugc.mp4"
    cover_p = "output/coffee_shop_ugc_cover.jpg"
    
    with open(temp_p, "wb") as f:
        f.write(content)
        
    print("Applying outro and cover...")
    apply_outro_and_cover(temp_p, final_p, cover_p)
    
    os.remove(temp_p)
    print(f"\nSuccess! \nVideo: {final_p}\nCover: {cover_p}")
except Exception as e:
    print(f"FAILED: {e}")
