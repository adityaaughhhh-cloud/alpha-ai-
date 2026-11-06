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
# ALPHA AI 5.2
# Chat + FLUX Image Generation + 720p Export
# Daily Quota + Supabase Storage
# Image Analysis + Image Tools + Video Analysis
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ALPHA")


# ============================================================
# 1. CONFIGURATION
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "alpha-images")

QUOTA_SALT = os.environ.get("ALPHA_QUOTA_SALT")

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gemini-3.5-flash-lite"
)

BACKUP_MODEL = os.environ.get(
    "BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)

DAILY_IMAGE_LIMIT = 10

IMAGE_TIMEOUT = int(os.environ.get("IMAGE_TIMEOUT", "60"))
VIDEO_TIMEOUT = int(os.environ.get("VIDEO_TIMEOUT", "120"))

PORT = int(os.environ.get("PORT", "10000"))

CHAT_HISTORY_LIMIT = 6

IST = ZoneInfo("Asia/Kolkata")

LOCAL_IMAGE_DIR = Path(
    os.environ.get("ALPHA_IMAGE_DIR", "/tmp/alpha-images")
)

LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ENVIRONMENT VALIDATION
# ============================================================

required_variables = {
    "GEMINI_API_KEY": API_KEY,
    "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
    "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "ALPHA_QUOTA_SALT": QUOTA_SALT
}

missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]

if missing_variables:
    raise RuntimeError(
        "Missing Render environment variables: "
        + ", ".join(missing_variables)
    )

if len(QUOTA_SALT) < 32:
    raise RuntimeError(
        "ALPHA_QUOTA_SALT must contain at least 32 characters."
    )


# ============================================================
# 3. API CLIENTS
# ============================================================

client = genai.Client(api_key=API_KEY)

http = requests.Session()

CLOUDFLARE_IMAGE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/"
    "@cf/black-forest-labs/flux-1-schnell"
)


# ============================================================
# 4. SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are Alpha AI.

Brand: Alpha AI
Tagline: Think Beyond Limits.

Be accurate, intelligent, useful and clear.

Never invent facts.
Admit uncertainty.
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
        r"\rightarrow": "→",
        r"\Rightarrow": "⇒"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("$$", "").replace("$", "")

    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1)/(\2)",
        text
    )

    return text.strip()


def get_error(error):

    logger.exception("ALPHA ERROR")

    message = str(error).lower()

    if "429" in message or "quota" in message:
        return (
            "API quota reached. "
            "Please try again later or check your API limits."
        )

    if "503" in message or "overloaded" in message:
        return (
            "The AI service is busy. "
            "Please try again shortly."
        )

    if "403" in message or "permission" in message:
        return (
            "API access denied. "
            "Please check your API permissions."
        )

    if "404" in message:
        return (
            "The selected model or API endpoint is unavailable."
        )

    if "timeout" in message or "timed out" in message:
        return (
            "The request timed out. "
            "Please try again."
        )

    if "401" in message or "unauthorized" in message:
        return (
            "API authentication failed. "
            "Check your environment variables."
        )

    return (
        "Alpha AI could not complete the request. "
        "Please try again."
    )


def get_daily_date():

    return datetime.now(IST).date().isoformat()


# ============================================================
# 6. VISITOR IDENTIFICATION
# ============================================================

def get_visitor_hash(request):

    ip = "unknown"

    try:

        if request and request.client:
            ip = request.client.host or "unknown"

        if request and request.headers:

            forwarded = request.headers.get(
                "x-forwarded-for"
            )

            if forwarded:
                ip = forwarded.split(",")[0].strip()

    except Exception:

        logger.warning(
            "Could not identify visitor IP."
        )

    return hashlib.sha256(
        (QUOTA_SALT + ":" + ip).encode("utf-8")
    ).hexdigest()


# ============================================================
# 7. SUPABASE DATABASE
# ============================================================

def supabase_headers():

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": (
            f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
        ),
        "Content-Type": "application/json"
    }


def get_image_usage(visitor_hash):

    today = get_daily_date()

    endpoint = (
        f"{SUPABASE_URL}/rest/v1/alpha_image_usage"
        f"?visitor_hash=eq.{visitor_hash}"
        f"&usage_date=eq.{today}"
        "&select=images_used"
    )

    response = requests.get(
        endpoint,
        headers=supabase_headers(),
        timeout=10
    )

    response.raise_for_status()

    rows = response.json()

    used = (
        rows[0]["images_used"]
        if rows
        else 0
    )

    return max(
        0,
        DAILY_IMAGE_LIMIT - used
    )


def reserve_image_credit(visitor_hash):

    endpoint = (
        f"{SUPABASE_URL}/rest/v1/rpc/"
        "consume_alpha_image_quota"
    )

    payload = {
        "p_visitor_hash": visitor_hash,
        "p_usage_date": get_daily_date(),
        "p_daily_limit": DAILY_IMAGE_LIMIT
    }

    response = requests.post(
        endpoint,
        headers=supabase_headers(),
        json=payload,
        timeout=10
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError(
            "Unexpected Supabase quota response."
        )

    return result


def refund_image_credit(visitor_hash, usage_date):

    endpoint = (
        f"{SUPABASE_URL}/rest/v1/rpc/"
        "refund_alpha_image_quota"
    )

    payload = {
        "p_visitor_hash": visitor_hash,
        "p_usage_date": usage_date
    }

    try:

        response = requests.post(
            endpoint,
            headers=supabase_headers(),
            json=payload,
            timeout=10
        )

        response.raise_for_status()

    except Exception:

        logger.exception(
            "IMAGE CREDIT REFUND FAILED"
        )


# ============================================================
# 8. CHAT ENGINE
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
                role=(
                    "user"
                    if role == "user"
                    else "model"
                ),
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
                "CHAT ERROR using %s",
                model
            )

            if answer:

                yield (
                    clean_text(answer)
                    + "\n\nConnection interrupted."
                )

                return

            if index == 1:

                yield get_error(error)

                return


# ============================================================
# 9. IMAGE CONVERSION
# ============================================================

def image_to_bytes(image_path, max_side=1800):

    with Image.open(image_path) as source:

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
            quality=88
        )

        return buffer.getvalue(), "image/jpeg"


# ============================================================
# 10. IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path, question):

    if not image_path:
        return "Please upload an image."

    question = (
        question or
        "Describe this image clearly."
    )

    try:

        image_bytes, mime = image_to_bytes(
            image_path
        )

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=[

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime
                ),

                types.Part.from_text(
                    text=question
                )
            ],

            config=types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                temperature=0.3,

                max_output_tokens=1500
            )
        )

        return clean_text(
            response.text or "No analysis generated."
        )

    except Exception as error:

        return get_error(error)


# ============================================================
# 11. IMAGE PROMPT ENHANCEMENT
# ============================================================

STYLE_GUIDANCE = {

    "Automatic": "",

    "Photorealistic":
        "Photorealistic appearance, realistic materials and natural lighting.",

    "Cinematic":
        "Cinematic framing, dramatic lighting and atmospheric depth.",

    "Anime":
        "High-quality anime illustration and polished character design.",

    "3D Render":
        "Detailed 3D rendering, realistic materials and studio lighting.",

    "Digital Art":
        "Detailed digital illustration and refined composition.",

    "Watercolor":
        "Traditional watercolor appearance and soft pigment textures.",

    "Oil Painting":
        "Oil painting appearance with layered brushwork.",

    "Pixel Art":
        "Clean pixel art and a controlled color palette.",

    "Minimalist":
        "Minimalist composition and clear visual hierarchy.",

    "Product Photography":
        "Professional product photography with controlled studio lighting."
}


def enhance_prompt(prompt, style, extra):

    prompt = (prompt or "").strip()

    if not prompt:

        raise ValueError(
            "Please describe the image you want."
        )

    sections = [

        "Create one original image based on this creative brief.",

        "User's description:",
        prompt,

        "Preserve the requested subject and important details.",

        "Use coherent composition, consistent perspective, "
        "believable lighting and carefully rendered details.",

        "Avoid unrelated objects, text, signatures and watermarks "
        "unless specifically requested."
    ]

    guidance = STYLE_GUIDANCE.get(
        style or "Automatic",
        ""
    )

    if guidance:

        sections.append(
            "Requested visual style: " + guidance
        )

    if extra and extra.strip():

        sections.append(
            "Additional instructions: " + extra.strip()
        )

    return "\n".join(sections)


# ============================================================
# 12. CLOUDFLARE FLUX IMAGE GENERATION
# ============================================================

def generate_flux_image(prompt):

    headers = {

        "Authorization":
            f"Bearer {CLOUDFLARE_API_TOKEN}",

        "Content-Type":
            "application/json"
    }

    # FLUX.1 Schnell API input.
    # Do not add unsupported width/height parameters.

    payload = {

        "prompt": prompt,

        "steps": 4
    }

    response = http.post(

        CLOUDFLARE_IMAGE_URL,

        headers=headers,

        json=payload,

        timeout=(10, IMAGE_TIMEOUT)
    )

    if not response.ok:

        raise RuntimeError(
            "Cloudflare image API returned HTTP "
            f"{response.status_code}: {response.text[:500]}"
        )

    data = response.json()

    if not data.get("success"):

        raise RuntimeError(
            "Cloudflare image generation failed."
        )

    image_data = (
        data.get("result", {}).get("image")
    )

    if not image_data:

        raise RuntimeError(
            "Cloudflare returned no image."
        )

    return base64.b64decode(image_data)


# ============================================================
# 13. 720P EXPORT ENGINE
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


def export_720p(image_bytes, ratio):

    width, height = RATIO_SIZES.get(
        ratio,
        (720, 720)
    )

    with Image.open(
        io.BytesIO(image_bytes)
    ) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        output = ImageOps.pad(

            image,

            (width, height),

            method=Image.Resampling.LANCZOS,

            color=(15, 15, 20),

            centering=(0.5, 0.5)
        )

        buffer = io.BytesIO()

        output.save(

            buffer,

            format="JPEG",

            quality=94,

            optimize=True
        )

        return buffer.getvalue(), output.copy()


# ============================================================
# 14. SUPABASE IMAGE STORAGE
# ============================================================

def store_image(image_bytes, image_id):

    safe_name = (
        datetime.now(IST).strftime("%Y/%m")
        + f"/{image_id}.jpg"
    )

    endpoint = (

        f"{SUPABASE_URL}/storage/v1/object/"
        f"{SUPABASE_BUCKET}/{safe_name}"
    )

    headers = {

        "Authorization":
            f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",

        "apikey":
            SUPABASE_SERVICE_ROLE_KEY,

        "Content-Type":
            "image/jpeg",

        "x-upsert":
            "true"
    }

    try:

        response = requests.post(

            endpoint,

            headers=headers,

            data=image_bytes,

            timeout=(5, 20)
        )

        response.raise_for_status()

        logger.info(
            "Image %s stored successfully.",
            image_id
        )

    except Exception:

        logger.exception(
            "SUPABASE STORAGE UPLOAD FAILED"
        )


def start_storage(image_bytes, image_id):

    thread = threading.Thread(

        target=store_image,

        args=(image_bytes, image_id),

        daemon=True
    )

    thread.start()


# ============================================================
# 15. MAIN IMAGE GENERATION
# ============================================================

def make_image_response(

    prompt,

    style,

    extra,

    ratio,

    request: gr.Request

):

    if not prompt or not prompt.strip():

        return (
            None,
            "Please enter an image description.",
            "10 / 10 remaining"
        )

    if request is None:

        return (
            None,
            "Unable to identify your visitor session.",
            "Quota status unavailable"
        )

    visitor_hash = get_visitor_hash(request)

    usage_date = get_daily_date()

    reserved = False

    start = time.monotonic()

    try:

        # Reserve one credit using an atomic database operation.

        quota = reserve_image_credit(
            visitor_hash
        )

        if not quota.get("allowed"):

            return (

                None,

                "### Daily image limit reached\n"
                "You have used all 10 free image generations. "
                "Your quota resets at midnight India time.",

                "0 / 10 remaining"
            )

        reserved = True

        enhanced_prompt = enhance_prompt(

            prompt,

            style,

            extra
        )

        logger.info(
            "Starting FLUX image generation."
        )

        image_bytes = generate_flux_image(
            enhanced_prompt
        )

        final_bytes, display_image = export_720p(

            image_bytes,

            ratio
        )

        image_id = uuid.uuid4().hex

        local_path = (
            LOCAL_IMAGE_DIR / f"{image_id}.jpg"
        )

        display_image.save(

            local_path,

            format="JPEG",

            quality=94
        )

        # Cloud storage runs in the background.

        start_storage(
            final_bytes,
            image_id
        )

        elapsed = time.monotonic() - start

        remaining = quota.get(
            "remaining",
            0
        )

        status = f"""
### Alpha AI Image Generator

- **Engine:** Cloudflare FLUX.1 Schnell
- **Export dimensions:** {display_image.width} × {display_image.height}
- **Aspect ratio:** {ratio}
- **Generation time:** {elapsed:.1f} seconds
- **Daily allowance:** 10 images
- **Images remaining:** {remaining}
- **Image ID:** `{image_id}`

Your image was generated successfully.
"""

        return (

            display_image,

            status,

            f"{remaining} / 10 remaining"
        )

    except Exception as error:

        logger.exception(
            "IMAGE GENERATION FAILED"
        )

        if reserved:

            refund_image_credit(

                visitor_hash,

                usage_date
            )

        try:

            remaining = get_image_usage(
                visitor_hash
            )

            quota_text = (
                f"{remaining} / 10 remaining"
            )

        except Exception:

            quota_text = (
                "Quota status unavailable"
            )

        return (

            None,

            get_error(error),

            quota_text
        )


# ============================================================
# 16. IMAGE RESIZING TOOL
# ============================================================

def resize_image_tool(image_path, scale):

    if not image_path:

        return None, "Upload an image first."

    try:

        factor = int(
            scale.replace("×", "").strip()
        )

        with Image.open(image_path) as source:

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

            resized = image.resize(

                (
                    image.width * factor,
                    image.height * factor
                ),

                Image.Resampling.LANCZOS
            )

            output = (
                LOCAL_IMAGE_DIR
                / f"resize-{uuid.uuid4().hex}.jpg"
            )

            resized.save(

                output,

                format="JPEG",

                quality=92
            )

            result = resized.copy()

        return (

            result,

            f"New dimensions: "
            f"{result.width} × {result.height}. "
            "Lanczos resizing does not reconstruct missing details."
        )

    except Exception as error:

        return None, get_error(error)


# ============================================================
# 17. VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path, question):

    if not video_path:

        return "Please upload a video."

    video_file = None

    start = time.monotonic()

    try:

        video_file = client.files.upload(
            file=video_path
        )

        processing_start = time.monotonic()

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

                return "Video processing failed."

            if (
                time.monotonic() - processing_start
                > VIDEO_TIMEOUT
            ):

                return "Video processing timed out."

            time.sleep(3)

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=[

                types.Part.from_uri(

                    file_uri=video_file.uri,

                    mime_type=video_file.mime_type
                ),

                types.Part.from_text(

                    text=(
                        question
                        or "Explain what happens in this video."
                    )
                )
            ],

            config=types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                temperature=0.3,

                max_output_tokens=1500
            )
        )

        elapsed = time.monotonic() - start

        return (

            clean_text(
                response.text or "No video analysis generated."
            )

            + f"\n\nAnalysis time: {elapsed:.1f} seconds."
        )

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
                    "VIDEO FILE CLEANUP FAILED"
                )


# ============================================================
# 18. USER INTERFACE
# ============================================================

CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

#alpha-header {
    text-align: center;
    padding: 20px;
}

#alpha-header h1 {
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 2px;
}

#alpha-header p {
    opacity: .75;
}

button {
    border-radius: 12px !important;
}

textarea {
    border-radius: 12px !important;
}
"""


with gr.Blocks(
    title="ALPHA AI 5.2"
) as app:

    gr.HTML("""
    <div id="alpha-header">
        <h1>🚀 ALPHA AI 5.2</h1>
        <p>THINK BEYOND LIMITS</p>
        <p>Your intelligent AI assistant.</p>
    </div>
    """)

    with gr.Tabs():

        # ====================================================
        # CHAT
        # ====================================================

        with gr.Tab("💬 Chat"):

            gr.ChatInterface(

                fn=alpha_chat,

                title="Alpha AI",

                description=(
                    "Ask questions and get intelligent answers."
                ),

                chatbot=gr.Chatbot(
                    height=500
                ),

                textbox=gr.Textbox(
                    placeholder="Message Alpha AI...",
                    submit_btn="Send"
                )
            )

        # ====================================================
        # IMAGE GENERATOR
        # ====================================================

        with gr.Tab("🎨 Image Generator"):

            gr.Markdown(
                "## Alpha AI Image Studio\n"
                "Create images using Cloudflare FLUX."
            )

            quota_display = gr.Markdown(
                "**Free image allowance: 10 images per day.**"
            )

            prompt_input = gr.Textbox(

                label="Describe your image",

                placeholder=(
                    "Example: A realistic view of Saturn "
                    "and its rings in deep space..."
                ),

                lines=4
            )

            with gr.Row():

                style_input = gr.Dropdown(

                    choices=list(
                        STYLE_GUIDANCE.keys()
                    ),

                    value="Automatic",

                    label="Visual style"
                )

                ratio_input = gr.Dropdown(

                    choices=list(
                        RATIO_SIZES.keys()
                    ),

                    value="16:9",

                    label="Aspect ratio"
                )

            extra_input = gr.Textbox(

                label="Additional instructions",

                placeholder=(
                    "Colors, lighting, composition..."
                ),

                lines=2
            )

            generate_button = gr.Button(

                "Generate Image",

                variant="primary"
            )

            generated_output = gr.Image(

                label="Generated image",

                type="pil"
            )

            generation_status = gr.Markdown()

            generate_button.click(

                fn=make_image_response,

                inputs=[

                    prompt_input,

                    style_input,

                    extra_input,

                    ratio_input
                ],

                outputs=[

                    generated_output,

                    generation_status,

                    quota_display
                ],

                api_name="generate_image"
            )

        # ====================================================
        # IMAGE TOOLS
        # ====================================================

        with gr.Tab("🛠️ Image Tools"):

            gr.Markdown(
                "## Image Resizing\n"
                "Resize images using Lanczos interpolation."
            )

            resize_input = gr.Image(

                label="Upload image",

                type="filepath"
            )

            scale_input = gr.Dropdown(

                choices=[
                    "2×",
                    "3×",
                    "4×"
                ],

                value="2×",

                label="Resize factor"
            )

            resize_button = gr.Button(

                "Resize Image",

                variant="primary"
            )

            resized_output = gr.Image(
                type="pil",
                label="Resized image"
            )

            resize_status = gr.Markdown()

            resize_button.click(

                fn=resize_image_tool,

                inputs=[
                    resize_input,
                    scale_input
                ],

                outputs=[
                    resized_output,
                    resize_status
                ]
            )

        # ====================================================
        # IMAGE ANALYSIS
        # ====================================================

        with gr.Tab("🖼️ Image Analysis"):

            gr.Markdown(
                "## Understand Images"
            )

            image_input = gr.Image(

                label="Upload image",

                type="filepath"
            )

            image_question = gr.Textbox(

                label="Ask Alpha",

                placeholder="What is in this image?"
            )

            image_button = gr.Button(

                "Analyze Image",

                variant="primary"
            )

            image_output = gr.Markdown()

            image_button.click(

                fn=analyze_image,

                inputs=[
                    image_input,
                    image_question
                ],

                outputs=image_output
            )

        # ====================================================
        # VIDEO ANALYSIS
        # ====================================================

        with gr.Tab("🎬 Video Analysis"):

            gr.Markdown(
                "## Understand Videos"
            )

            video_input = gr.Video(

                label="Upload video",

                sources=["upload"]
            )

            video_question = gr.Textbox(

                label="Ask Alpha",

                placeholder=(
                    "Explain what happens in this video."
                )
            )

            video_button = gr.Button(

                "Analyze Video",

                variant="primary"
            )

            video_output = gr.Markdown()

            video_button.click(

                fn=analyze_video,

                inputs=[
                    video_input,
                    video_question
                ],

                outputs=video_output
            )

    gr.Markdown("""
    ---
    **ALPHA AI 5.2** | Think Beyond Limits

    Powered by Google Gemini and Cloudflare Workers AI.
    """)


# ============================================================
# 19. QUEUE AND APPLICATION START
# ============================================================

app.queue(

    max_size=20,

    default_concurrency_limit=2
)


if __name__ == "__main__":

    logger.info(
        "STARTING ALPHA AI 5.2"
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        theme=gr.themes.Soft(

            primary_hue="blue",

            secondary_hue="slate"
        ),

        css=CSS,

        show_error=False
        )
