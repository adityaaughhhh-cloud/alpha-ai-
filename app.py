import os
import io
import re
import time
import uuid
import base64
import hashlib
import logging
import threading

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import requests

from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI 6.0
# CHAT + IMAGE + VIDEO GENERATION
# 10 IMAGES + 3 VIDEOS DAILY
# 720P IMAGE AND VIDEO OUTPUT
# SUPABASE QUOTA + STORAGE
# ============================================================


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ALPHA")


# ============================================================
# 1. CONFIGURATION
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

CLOUDFLARE_API_TOKEN = os.environ.get(
    "CLOUDFLARE_API_TOKEN"
)

CLOUDFLARE_ACCOUNT_ID = os.environ.get(
    "CLOUDFLARE_ACCOUNT_ID"
)

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", ""
).rstrip("/")

SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY"
)

SUPABASE_BUCKET = os.environ.get(
    "SUPABASE_BUCKET", "alpha-media"
)

QUOTA_SALT = os.environ.get(
    "ALPHA_QUOTA_SALT"
)


CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gemini-3.5-flash-lite"
)

BACKUP_MODEL = os.environ.get(
    "BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)

VIDEO_MODEL = os.environ.get(
    "VIDEO_MODEL",
    "gemini-omni-1.1-flash"
)


DAILY_IMAGE_LIMIT = 10
DAILY_VIDEO_LIMIT = 3


IMAGE_TIMEOUT = 90
VIDEO_TIMEOUT = 600

PORT = int(
    os.environ.get("PORT", "10000")
)


CHAT_HISTORY_LIMIT = 8

IST = ZoneInfo("Asia/Kolkata")


LOCAL_DIR = Path(
    os.environ.get(
        "ALPHA_MEDIA_DIR",
        "/tmp/alpha-media"
    )
)

LOCAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ENVIRONMENT VALIDATION
# ============================================================

required_variables = {

    "GEMINI_API_KEY": API_KEY,

    "CLOUDFLARE_API_TOKEN":
        CLOUDFLARE_API_TOKEN,

    "CLOUDFLARE_ACCOUNT_ID":
        CLOUDFLARE_ACCOUNT_ID,

    "SUPABASE_URL":
        SUPABASE_URL,

    "SUPABASE_SERVICE_ROLE_KEY":
        SUPABASE_SERVICE_ROLE_KEY,

    "ALPHA_QUOTA_SALT":
        QUOTA_SALT
}


missing = [

    key

    for key, value in required_variables.items()

    if not value

]


if missing:

    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing)
    )


if len(QUOTA_SALT) < 32:

    raise RuntimeError(
        "ALPHA_QUOTA_SALT must contain "
        "at least 32 characters."
    )


# ============================================================
# 3. API CLIENTS
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


http = requests.Session()


CLOUDFLARE_IMAGE_URL = (

    "https://api.cloudflare.com/client/v4/accounts/"

    f"{CLOUDFLARE_ACCOUNT_ID}"

    "/ai/run/"

    "@cf/black-forest-labs/flux-1-schnell"

)


# ============================================================
# 4. SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """

You are Alpha AI.

Company: Alpha AI.

Tagline: Think Beyond Limits.

You are an intelligent and helpful AI assistant.

Your priorities:

1. Accuracy.
2. Clear explanations.
3. Useful answers.
4. Logical reasoning.
5. User safety.
6. Honest communication.

Never invent information.

Admit uncertainty when necessary.

Use conversation history when relevant.

Explain calculations step by step when requested.

Use understandable language.

"""


# ============================================================
# 5. GENERAL HELPERS
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    replacements = {

        r"\times": "×",

        r"\cdot": "·",

        r"\div": "÷",

        r"\pm": "±",

        r"\neq": "≠",

        r"\leq": "≤",

        r"\geq": "≥",

        r"\approx": "≈",

        r"\pi": "π",

        r"\infty": "∞",

        r"\rightarrow": "→"

    }

    for old, new in replacements.items():

        text = text.replace(old, new)

    text = text.replace("$$", "").replace("$", "")

    return text.strip()


def get_error(error):

    logger.exception("ALPHA ERROR")

    message = str(error).lower()

    if "429" in message or "quota" in message:

        return "API quota reached. Please try later."

    if "503" in message or "overloaded" in message:

        return "AI service is busy. Try again shortly."

    if "403" in message:

        return "API permission denied. Check API access."

    if "401" in message:

        return "API authentication failed."

    if "404" in message:

        return "Requested AI model is unavailable."

    if "timeout" in message:

        return "Request timed out. Please try again."

    return (
        "Alpha AI could not complete your request. "
        "Please try again."
    )


def today():

    return datetime.now(IST).date().isoformat()


# ============================================================
# 6. VISITOR IDENTIFICATION
# ============================================================

def visitor_hash(request):

    ip = "unknown"

    try:

        if request and request.client:

            ip = request.client.host or "unknown"

        # Only use forwarded IP headers when the
        # deployment proxy is configured to trust them.

    except Exception:

        logger.warning("Visitor identification failed.")

    return hashlib.sha256(

        (
            QUOTA_SALT + ":" + ip
        ).encode()

    ).hexdigest()


# ============================================================
# 7. SUPABASE HELPERS
# ============================================================

def supabase_headers():

    return {

        "apikey": SUPABASE_SERVICE_ROLE_KEY,

        "Authorization":
            f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",

        "Content-Type": "application/json"

    }


def consume_quota(visitor, media_type):

    endpoint = (

        f"{SUPABASE_URL}/rest/v1/rpc/"

        "consume_alpha_media_quota"

    )

    limit = (

        DAILY_IMAGE_LIMIT

        if media_type == "image"

        else DAILY_VIDEO_LIMIT

    )

    payload = {

        "p_visitor_hash": visitor,

        "p_usage_date": today(),

        "p_media_type": media_type,

        "p_daily_limit": limit

    }

    response = requests.post(

        endpoint,

        headers=supabase_headers(),

        json=payload,

        timeout=15

    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):

        raise RuntimeError(
            "Invalid quota response."
        )

    return result


def refund_quota(visitor, media_type, usage_date):

    endpoint = (

        f"{SUPABASE_URL}/rest/v1/rpc/"

        "refund_alpha_media_quota"

    )

    payload = {

        "p_visitor_hash": visitor,

        "p_usage_date": usage_date,

        "p_media_type": media_type

    }

    try:

        response = requests.post(

            endpoint,

            headers=supabase_headers(),

            json=payload,

            timeout=15

        )

        response.raise_for_status()

    except Exception:

        logger.exception("QUOTA REFUND FAILED")


# ============================================================
# 8. QUOTA DISPLAY
# ============================================================

def quota_status(visitor):

    image_used = consume_quota_status(
        visitor,
        "image",
        DAILY_IMAGE_LIMIT
    )

    video_used = consume_quota_status(
        visitor,
        "video",
        DAILY_VIDEO_LIMIT
    )

    return f"""

### Your Daily Free Allowance

| Feature | Remaining |
|---|---:|
| Images | {image_used} / 10 |
| Videos | {video_used} / 3 |

Resets at midnight India time.

"""


def consume_quota_status(visitor, media_type, limit):

    endpoint = (

        f"{SUPABASE_URL}/rest/v1/alpha_media_usage"

        f"?visitor_hash=eq.{visitor}"

        f"&usage_date=eq.{today()}"

        f"&media_type=eq.{media_type}"

        "&select=used"

    )

    response = requests.get(

        endpoint,

        headers=supabase_headers(),

        timeout=10

    )

    response.raise_for_status()

    rows = response.json()

    used = rows[0]["used"] if rows else 0

    return max(0, limit - used)


# ============================================================
# 9. CHAT ENGINE
# ============================================================

def prepare_history(history):

    contents = []

    for item in (history or [])[-CHAT_HISTORY_LIMIT:]:

        if not isinstance(item, dict):

            continue

        role = item.get("role")

        content = item.get("content")

        if role not in ("user", "assistant"):

            continue

        if not isinstance(content, str):

            continue

        if not content.strip():

            continue

        contents.append(

            types.Content(

                role="user" if role == "user" else "model",

                parts=[
                    types.Part.from_text(text=content)
                ]

            )

        )

    return contents


def alpha_chat(message, history):

    if not message or not message.strip():

        yield "Please enter a message."

        return

    contents = prepare_history(history)

    contents.append(

        types.Content(

            role="user",

            parts=[

                types.Part.from_text(
                    text=message.strip()
                )

            ]

        )

    )

    config = types.GenerateContentConfig(

        system_instruction=SYSTEM_INSTRUCTION,

        temperature=0.3,

        max_output_tokens=2048

    )

    for index, model in enumerate(
        [CHAT_MODEL, BACKUP_MODEL]
    ):

        answer = ""

        try:

            stream = client.models.generate_content_stream(

                model=model,

                contents=contents,

                config=config

            )

            for chunk in stream:

                if chunk.text:

                    answer += chunk.text

                    yield clean_text(answer)

            if answer:

                return

            yield "No response generated."

            return

        except Exception as error:

            logger.exception(
                "CHAT ERROR: %s",
                model
            )

            if answer:

                yield clean_text(answer)

                return

            if index == 1:

                yield get_error(error)

                return


# ============================================================
# 10. IMAGE CONVERSION
# ============================================================

def image_to_bytes(path, max_side=1800):

    with Image.open(path) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        image.thumbnail(
            (max_side, max_side)
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90
        )

        return buffer.getvalue(), "image/jpeg"


# ============================================================
# 11. IMAGE ANALYSIS
# ============================================================

def analyze_image(path, question):

    if not path:

        return "Upload an image first."

    try:

        data, mime = image_to_bytes(path)

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=[

                types.Part.from_bytes(

                    data=data,

                    mime_type=mime

                ),

                types.Part.from_text(

                    text=question or "Describe this image."

                )

            ],

            config=types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                max_output_tokens=1500

            )

        )

        return clean_text(
            response.text
        )

    except Exception as error:

        return get_error(error)


# ============================================================
# 12. IMAGE GENERATION
# ============================================================

STYLE_GUIDANCE = {

    "Automatic": "",

    "Photorealistic":
        "Photorealistic details and natural lighting.",

    "Cinematic":
        "Cinematic composition and dramatic lighting.",

    "Anime":
        "High-quality anime illustration.",

    "3D Render":
        "Detailed professional 3D rendering.",

    "Digital Art":
        "Detailed digital artwork.",

    "Watercolor":
        "Traditional watercolor painting.",

    "Oil Painting":
        "Detailed oil painting.",

    "Pixel Art":
        "Clean pixel art.",

    "Minimalist":
        "Minimalist visual composition.",

    "Product Photography":
        "Professional studio product photography."

}


def enhance_prompt(prompt, style, extra):

    guidance = STYLE_GUIDANCE.get(
        style,
        ""
    )

    result = (

        "Create one original high-quality image.\n"

        "User description: "

        + prompt.strip()

        + "\n"

        + guidance

        + "\n"

        + (extra or "")

    )

    return result[:4000]


def generate_flux_image(prompt):

    payload = {

        "prompt": prompt[:2048],

        "steps": 4

    }

    headers = {

        "Authorization":
            f"Bearer {CLOUDFLARE_API_TOKEN}",

        "Content-Type":
            "application/json"

    }

    response = http.post(

        CLOUDFLARE_IMAGE_URL,

        headers=headers,

        json=payload,

        timeout=(10, IMAGE_TIMEOUT)

    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):

        raise RuntimeError(
            "Cloudflare image generation failed."
        )

    encoded = data.get(
        "result", {}
    ).get("image")

    if not encoded:

        raise RuntimeError(
            "No image received."
        )

    return base64.b64decode(encoded)


# ============================================================
# 13. 720P IMAGE EXPORT
# ============================================================

RATIO_SIZES = {

    "1:1": (720, 720),

    "16:9": (1280, 720),

    "9:16": (720, 1280),

    "4:3": (960, 720),

    "3:4": (720, 960),

    "3:2": (1080, 720),

    "2:3": (720, 1080),

    "21:9": (1680, 720)

}


def export_image(data, ratio):

    width, height = RATIO_SIZES.get(
        ratio,
        (1280, 720)
    )

    with Image.open(
        io.BytesIO(data)
    ) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        result = ImageOps.pad(

            image,

            (width, height),

            method=Image.Resampling.LANCZOS,

            color=(15, 15, 20)

        )

        buffer = io.BytesIO()

        result.save(

            buffer,

            format="JPEG",

            quality=94

        )

        return buffer.getvalue(), result.copy()


# ============================================================
# 14. STORAGE
# ============================================================

def store_media(data, filename, mime):

    endpoint = (

        f"{SUPABASE_URL}/storage/v1/object/"

        f"{SUPABASE_BUCKET}/{filename}"

    )

    headers = {

        "Authorization":
            f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",

        "apikey":
            SUPABASE_SERVICE_ROLE_KEY,

        "Content-Type": mime,

        "x-upsert": "true"

    }

    try:

        response = requests.post(

            endpoint,

            headers=headers,

            data=data,

            timeout=(5, 30)

        )

        response.raise_for_status()

        logger.info(
            "Stored media: %s",
            filename
        )

    except Exception:

        logger.exception(
            "MEDIA STORAGE FAILED"
        )


def background_storage(data, filename, mime):

    threading.Thread(

        target=store_media,

        args=(data, filename, mime),

        daemon=True

    ).start()


# ============================================================
# 15. IMAGE GENERATION HANDLER
# ============================================================

def make_image(

    prompt,

    style,

    extra,

    ratio,

    request: gr.Request

):

    if not prompt or not prompt.strip():

        return None, "Enter an image description.", None

    if request is None:

        return None, "Visitor session unavailable.", None

    visitor = visitor_hash(request)

    usage_date = today()

    reserved = False

    start = time.monotonic()

    try:

        quota = consume_quota(
            visitor,
            "image"
        )

        if not quota.get("allowed"):

            return (

                None,

                "Daily limit reached: 10 images.",

                None

            )

        reserved = True

        enhanced = enhance_prompt(

            prompt,

            style,

            extra

        )

        raw = generate_flux_image(enhanced)

        final_bytes, display = export_image(
            raw,
            ratio
        )

        image_id = uuid.uuid4().hex

        output_path = LOCAL_DIR / f"{image_id}.jpg"

        display.save(
            output_path,
            format="JPEG",
            quality=94
        )

        background_storage(

            final_bytes,

            f"images/{image_id}.jpg",

            "image/jpeg"

        )

        elapsed = time.monotonic() - start

        remaining = quota.get("remaining", 0)

        status = f"""

### Image generated successfully!

- Engine: Cloudflare FLUX.1 Schnell
- Resolution: {display.width} × {display.height}
- Aspect ratio: {ratio}
- Time: {elapsed:.1f} seconds
- Remaining images: {remaining}
- Image ID: `{image_id}`

"""

        return (

            display,

            status,

            str(output_path)

        )

    except Exception as error:

        if reserved:

            refund_quota(
                visitor,
                "image",
                usage_date
            )

        return None, get_error(error), None


# ============================================================
# 16. VIDEO GENERATION ENGINE
# ============================================================

def generate_video(

    prompt,

    image_path,

    ratio,

    request: gr.Request

):

    if not prompt or not prompt.strip():

        return None, "Enter a video description."

    if request is None:

        return None, "Visitor session unavailable."

    visitor = visitor_hash(request)

    usage_date = today()

    reserved = False

    started = time.monotonic()

    try:

        quota = consume_quota(
            visitor,
            "video"
        )

        if not quota.get("allowed"):

            return (

                None,

                "### Daily video limit reached\n"
                "You have used all 3 free video generations."

            )

        reserved = True

        # TEXT TO VIDEO

        if not image_path:

            video_input = (

                "Create a high-quality cinematic video: "

                + prompt.strip()

            )

            task = "text_to_video"

        # IMAGE TO VIDEO

        else:

            image_data, mime = image_to_bytes(
                image_path
            )

            encoded = base64.b64encode(
                image_data
            ).decode("utf-8")

            video_input = [

                {

                    "type": "image",

                    "data": encoded,

                    "mime_type": mime

                },

                {

                    "type": "text",

                    "text": (
                        "Animate this image into a "
                        "cinematic video. Preserve its "
                        "main subject and visual identity. "
                        + prompt.strip()
                    )

                }

            ]

            task = "image_to_video"

        # GEMINI OMNI INTERACTIONS API

        interaction = client.interactions.create(

            model=VIDEO_MODEL,

            input=video_input,

            response_format={

                "type": "video",

                "resolution": "720p",

                "aspect_ratio": ratio

            },

            generation_config={

                "video_config": {

                    "task": task

                }

            }

        )

        video_data = getattr(
            interaction,
            "output_video",
            None
        )

        if not video_data or not video_data.data:

            raise RuntimeError(
                "Video generation returned no video data."
            )

        video_bytes = base64.b64decode(
            video_data.data
        )

        video_id = uuid.uuid4().hex

        output_path = LOCAL_DIR / f"{video_id}.mp4"

        with open(output_path, "wb") as file:

            file.write(video_bytes)

        background_storage(

            video_bytes,

            f"videos/{video_id}.mp4",

            "video/mp4"

        )

        elapsed = time.monotonic() - started

        remaining = quota.get(
            "remaining",
            0
        )

        status = f"""

### Video generated successfully!

- Engine: Gemini Omni Flash
- Mode: {task}
- Resolution: 720p
- Aspect ratio: {ratio}
- Time: {elapsed:.1f} seconds
- Remaining videos: {remaining}
- Video ID: `{video_id}`

Your MP4 video is ready to download.

"""

        return str(output_path), status

    except Exception as error:

        logger.exception(
            "VIDEO GENERATION FAILED"
        )

        if reserved:

            refund_quota(

                visitor,

                "video",

                usage_date

            )

        return None, get_error(error)


# ============================================================
# 17. IMAGE RESIZING
# ============================================================

def resize_image(path, scale):

    if not path:

        return None, "Upload an image first."

    try:

        factor = int(
            scale.replace("×", "")
        )

        with Image.open(path) as source:

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

            result = image.resize(

                (

                    image.width * factor,

                    image.height * factor

                ),

                Image.Resampling.LANCZOS

            )

            return (

                result,

                f"New dimensions: "
                f"{result.width} × {result.height}"

            )

    except Exception as error:

        return None, get_error(error)


# ============================================================
# 18. VIDEO ANALYSIS
# ============================================================

def analyze_video(path, question):

    if not path:

        return "Upload a video first."

    video_file = None

    try:

        video_file = client.files.upload(
            file=path
        )

        started = time.monotonic()

        while True:

            video_file = client.files.get(
                name=video_file.name
            )

            state = getattr(
                video_file.state,
                "name",
                str(video_file.state)
            ).upper()

            if state == "ACTIVE":

                break

            if state == "FAILED":

                raise RuntimeError(
                    "Video processing failed."
                )

            if time.monotonic() - started > VIDEO_TIMEOUT:

                raise TimeoutError(
                    "Video processing timed out."
                )

            time.sleep(3)

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=[

                types.Part.from_uri(

                    file_uri=video_file.uri,

                    mime_type=video_file.mime_type

                ),

                types.Part.from_text(

                    text=question or "Describe this video."

                )

            ],

            config=types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                max_output_tokens=1500

            )

        )

        return clean_text(response.text)

    except Exception as error:

        return get_error(error)

    finally:

        if video_file:

            try:

                client.files.delete(
                    name=video_file.name
                )

            except Exception:

                logger.warning(
                    "Video cleanup failed."
                )


# ============================================================
# 19. INTERFACE DESIGN
# ============================================================

CSS = """

.gradio-container {

    max-width: 1300px !important;

    margin: auto !important;

}

#alpha-header {

    text-align: center;

    padding: 25px;

}

#alpha-header h1 {

    font-size: 42px;

    font-weight: 900;

    letter-spacing: 2px;

}

#alpha-header p {

    opacity: .8;

}

button {

    border-radius: 12px !important;

}

textarea {

    border-radius: 12px !important;

}

"""


# ============================================================
# 20. GRADIO APPLICATION
# ============================================================

with gr.Blocks(

    title="ALPHA AI 6.0",

    theme=gr.themes.Soft(

        primary_hue="blue",

        secondary_hue="slate"

    ),

    css=CSS

) as app:

    gr.HTML("""

    <div id="alpha-header">

        <h1>🚀 ALPHA AI 6.0</h1>

        <p>THINK BEYOND LIMITS</p>

        <p>Your intelligent AI platform.</p>

    </div>

    """)

    with gr.Tabs():

        # ----------------------------------------------------
        # CHAT
        # ----------------------------------------------------

        with gr.Tab("💬 Chat"):

            gr.ChatInterface(

                fn=alpha_chat,

                title="Alpha AI",

                description="Your intelligent AI assistant.",

                chatbot=gr.Chatbot(height=500),

                textbox=gr.Textbox(

                    placeholder="Message Alpha AI...",

                    submit_btn="Send"

                )

            )

        # ----------------------------------------------------
        # IMAGE GENERATOR
        # ----------------------------------------------------

        with gr.Tab("🎨 Image Generator"):

            gr.Markdown(
                "## Alpha AI Image Studio"
            )

            gr.Markdown(
                "**10 free image generations every day.**"
            )

            prompt = gr.Textbox(

                label="Image description",

                placeholder="Describe your image...",

                lines=4

            )

            with gr.Row():

                style = gr.Dropdown(

                    choices=list(STYLE_GUIDANCE.keys()),

                    value="Automatic",

                    label="Visual style"

                )

                ratio = gr.Dropdown(

                    choices=list(RATIO_SIZES.keys()),

                    value="16:9",

                    label="Aspect ratio"

                )

            extra = gr.Textbox(

                label="Additional instructions",

                lines=2

            )

            image_button = gr.Button(

                "🚀 Generate Image",

                variant="primary"

            )

            image_output = gr.Image(
                type="pil",
                label="Generated image"
            )

            image_status = gr.Markdown()

            image_download = gr.File(
                label="Download image"
            )

            image_button.click(

                fn=make_image,

                inputs=[

                    prompt,

                    style,

                    extra,

                    ratio

                ],

                outputs=[

                    image_output,

                    image_status,

                    image_download

                ],

                api_name="generate_image"

            )

        # ----------------------------------------------------
        # VIDEO GENERATOR
        # ----------------------------------------------------

        with gr.Tab("🎬 Video Generator"):

            gr.Markdown(
                "## Alpha AI Video Studio"
            )

            gr.Markdown(
                "**3 free video generations every day.**"
            )

            video_prompt = gr.Textbox(

                label="Describe your video",

                placeholder=(
                    "A cinematic shot of Saturn "
                    "rotating in deep space..."
                ),

                lines=4

            )

            video_image = gr.Image(

                label="Optional starting image",

                type="filepath"

            )

            gr.Markdown(
                "Leave the image empty for text-to-video. "
                "Upload an image for image-to-video."
            )

            video_ratio = gr.Dropdown(

                choices=["16:9", "9:16"],

                value="16:9",

                label="Video aspect ratio"

            )

            video_button = gr.Button(

                "🎥 Generate 720p Video",

                variant="primary"

            )

            video_output = gr.Video(

                label="Generated video",

                format="mp4"

            )

            video_status = gr.Markdown()

            video_button.click(

                fn=generate_video,

                inputs=[

                    video_prompt,

                    video_image,

                    video_ratio

                ],

                outputs=[

                    video_output,

                    video_status

                ],

                api_name="generate_video"

            )

        # ----------------------------------------------------
        # IMAGE TOOLS
        # ----------------------------------------------------

        with gr.Tab("🛠️ Image Tools"):

            resize_input = gr.Image(

                type="filepath",

                label="Upload image"

            )

            resize_scale = gr.Dropdown(

                choices=["2×", "3×", "4×"],

                value="2×",

                label="Resize factor"

            )

            resize_button = gr.Button(
                "Resize Image"
            )

            resize_output = gr.Image(
                type="pil"
            )

            resize_status = gr.Markdown()

            resize_button.click(

                fn=resize_image,

                inputs=[

                    resize_input,

                    resize_scale

                ],

                outputs=[

                    resize_output,

                    resize_status

                ]

            )

        # ----------------------------------------------------
        # IMAGE ANALYSIS
        # ----------------------------------------------------

        with gr.Tab("🖼️ Image Analysis"):

            analysis_image = gr.Image(

                type="filepath",

                label="Upload image"

            )

            analysis_question = gr.Textbox(

                label="Ask Alpha",

                placeholder="What is in this image?"

            )

            analysis_button = gr.Button(
                "Analyze Image"
            )

            analysis_output = gr.Markdown()

            analysis_button.click(

                fn=analyze_image,

                inputs=[

                    analysis_image,

                    analysis_question

                ],

                outputs=analysis_output

            )

        # ----------------------------------------------------
        # VIDEO ANALYSIS
        # ----------------------------------------------------

        with gr.Tab("📹 Video Analysis"):

            analysis_video = gr.Video(

                sources=["upload"],

                label="Upload video"

            )

            video_question = gr.Textbox(

                label="Ask Alpha",

                placeholder="Explain this video."

            )

            analysis_button = gr.Button(
                "Analyze Video"
            )

            analysis_output = gr.Markdown()

            analysis_button.click(

                fn=analyze_video,

                inputs=[

                    analysis_video,

                    video_question

                ],

                outputs=analysis_output

            )

    gr.Markdown("""

    ---

    # ALPHA AI 6.0

    **THINK BEYOND LIMITS**

    Powered by Google Gemini and Cloudflare FLUX.

    """)


# ============================================================
# 21. QUEUE
# ============================================================

app.queue(

    max_size=20,

    default_concurrency_limit=2

)


# ============================================================
# 22. START APPLICATION
# ============================================================

if __name__ == "__main__":

    logger.info(
        "STARTING ALPHA AI 6.0"
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        show_error=False

    )
