import os
import io
import uuid
import time
import base64
import hashlib
import logging
import ast
import operator
import re
import threading

from collections import OrderedDict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import requests

from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI 16.0 - SPEED OPTIMIZED
#
# TEXT CHAT + IMAGE STUDIO + IMAGE ANALYSIS + VIDEO ANALYSIS
#
# Chat improvements:
# - Primary model only
# - Shorter conversation history
# - Lower output token limits
# - Streaming responses
# - Local calculator
# - Response cache
# ============================================================


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALPHA")


# ============================================================
# CONFIGURATION
# ============================================================

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", ""
).rstrip("/")

SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY"
)

QUOTA_SALT = os.environ.get("ALPHA_QUOTA_SALT")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# Gemini models

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

GEMINI_BACKUP_MODEL = os.environ.get(
    "GEMINI_BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)


# ============================================================
# SPEED SETTINGS
# ============================================================

# Chat request timeout in milliseconds.
CHAT_TIMEOUT = 10000

# Media operations retain a longer timeout.
API_TIMEOUT = 120000

# Image generation timeout.
IMAGE_TIMEOUT = 120

# Daily image generation limit.
DAILY_IMAGE_LIMIT = 10

# Maximum video upload size.
VIDEO_MAX_SIZE = 50 * 1024 * 1024

PORT = int(os.environ.get("PORT", "10000"))

IST = ZoneInfo("Asia/Kolkata")

OUTPUT_DIR = Path("/tmp/alpha-images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

required = {
    "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
    "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "ALPHA_QUOTA_SALT": QUOTA_SALT,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing = [
    key for key, value in required.items()
    if not value
]

if missing:
    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing)
    )

if len(QUOTA_SALT) < 32:
    raise RuntimeError(
        "ALPHA_QUOTA_SALT must contain at least 32 characters."
    )


# ============================================================
# API CLIENTS
# ============================================================

http = requests.Session()

# Dedicated fast Chat client.
chat_gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=CHAT_TIMEOUT
    ),
)

# Media client with longer timeout.
gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=API_TIMEOUT
    ),
)


CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    + CLOUDFLARE_ACCOUNT_ID
    + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
)


# ============================================================
# RESPONSE CACHE
# ============================================================

CACHE_TTL_SECONDS = 1800
CACHE_MAX_ITEMS = 128

_answer_cache = OrderedDict()
_cache_lock = threading.Lock()


def cache_get(key):

    now = time.monotonic()

    with _cache_lock:

        item = _answer_cache.get(key)

        if not item:
            return None

        created, value = item

        if now - created > CACHE_TTL_SECONDS:

            _answer_cache.pop(key, None)

            return None

        _answer_cache.move_to_end(key)

        return value


def cache_put(key, value):

    with _cache_lock:

        _answer_cache[key] = (
            time.monotonic(),
            value
        )

        _answer_cache.move_to_end(key)

        while len(_answer_cache) > CACHE_MAX_ITEMS:

            _answer_cache.popitem(last=False)


def cache_key(message, history):

    recent = history[-4:]

    raw = (
        repr([
            (x["role"], x["content"])
            for x in recent
        ])
        + "\n"
        + message
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# COMMON HELPERS
# ============================================================

def today():

    return datetime.now(IST).date().isoformat()


def get_visitor(request):

    ip = "unknown"

    try:

        if request and request.client:

            ip = request.client.host or "unknown"

    except Exception:

        logger.exception(
            "VISITOR IDENTIFICATION ERROR"
        )

    return hashlib.sha256(
        (QUOTA_SALT + ":" + ip).encode("utf-8")
    ).hexdigest()


def supabase_headers():

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": (
            "Bearer " + SUPABASE_SERVICE_ROLE_KEY
        ),
        "Content-Type": "application/json",
    }


# ============================================================
# SUPABASE IMAGE QUOTA
# ============================================================

def consume_image_quota(visitor):

    endpoint = (
        SUPABASE_URL
        + "/rest/v1/rpc/consume_alpha_media_quota"
    )

    payload = {
        "p_visitor_hash": visitor,
        "p_usage_date": today(),
        "p_media_type": "image",
        "p_daily_limit": DAILY_IMAGE_LIMIT,
    }

    response = http.post(
        endpoint,
        headers=supabase_headers(),
        json=payload,
        timeout=20,
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict) or "allowed" not in result:

        raise RuntimeError(
            "Invalid image quota response."
        )

    return result


def refund_image_quota(visitor, usage_date):

    endpoint = (
        SUPABASE_URL
        + "/rest/v1/rpc/refund_alpha_media_quota"
    )

    payload = {
        "p_visitor_hash": visitor,
        "p_usage_date": usage_date,
        "p_media_type": "image",
    }

    try:

        response = http.post(
            endpoint,
            headers=supabase_headers(),
            json=payload,
            timeout=20,
        )

        response.raise_for_status()

    except Exception:

        logger.exception(
            "IMAGE CREDIT REFUND FAILED"
        )


# ============================================================
# IMAGE SETTINGS
# ============================================================

STYLES = {

    "Automatic": "",

    "Photorealistic":
        "Photorealistic image, realistic lighting and details.",

    "Cinematic":
        "Cinematic composition and dramatic lighting.",

    "Anime":
        "High quality anime illustration.",

    "3D Render":
        "Professional detailed 3D render.",

    "Digital Art":
        "Professional digital artwork.",

    "Watercolor":
        "Traditional watercolor painting.",

    "Oil Painting":
        "Detailed oil painting.",

    "Pixel Art":
        "High quality pixel art.",

    "Minimalist":
        "Clean minimalist artistic composition.",

    "Product Photography":
        "Professional studio product photography.",
}


RATIOS = {

    "1:1": (720, 720),

    "16:9": (1280, 720),

    "9:16": (720, 1280),

    "4:3": (960, 720),

    "3:4": (720, 960),

    "3:2": (1080, 720),

    "2:3": (720, 1080),

    "21:9": (1680, 720),
}


def build_prompt(prompt, style="Automatic", extra=""):

    return (
        prompt.strip()
        + "\n"
        + STYLES.get(style, "")
        + "\n"
        + (extra or "")
    )[:2048]


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt):

    headers = {

        "Authorization": (
            "Bearer " + CLOUDFLARE_API_TOKEN
        ),

        "Content-Type": "application/json",
    }

    response = http.post(

        CLOUDFLARE_URL,

        headers=headers,

        json={
            "prompt": prompt[:2048],
            "steps": 4,
        },

        timeout=(10, IMAGE_TIMEOUT),
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "application/json" in content_type:

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
                "No image returned by Cloudflare."
            )

        image_bytes = base64.b64decode(encoded)

    else:

        image_bytes = response.content

    with Image.open(
        io.BytesIO(image_bytes)
    ) as image:

        image.verify()

    return image_bytes


def export_image(image_bytes, ratio="16:9"):

    width, height = RATIOS.get(
        ratio,
        (1280, 720)
    )

    with Image.open(
        io.BytesIO(image_bytes)
    ) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        result = ImageOps.pad(

            image,

            (width, height),

            method=Image.Resampling.LANCZOS,

            color=(15, 15, 20),
        )

        output = (
            OUTPUT_DIR
            / (uuid.uuid4().hex + ".jpg")
        )

        result.save(
            output,
            format="JPEG",
            quality=94,
        )

        return str(output)


def create_image(
    prompt,
    request,
    ratio="16:9",
    style="Automatic",
    extra=""
):

    if not prompt or not prompt.strip():

        raise ValueError(
            "Enter an image description."
        )

    if request is None:

        raise RuntimeError(
            "Request information unavailable."
        )

    visitor = get_visitor(request)

    usage_date = today()

    reserved = False

    try:

        quota = consume_image_quota(visitor)

        if not quota.get("allowed"):

            raise RuntimeError(
                "Daily image limit reached. "
                "You can create 10 images per day."
            )

        reserved = True

        image_bytes = generate_image(
            build_prompt(prompt, style, extra)
        )

        return export_image(
            image_bytes,
            ratio
        )

    except Exception:

        if reserved:

            refund_image_quota(
                visitor,
                usage_date
            )

        raise


# ============================================================
# LOCAL CALCULATOR
# ============================================================

FAST_OPERATORS = {

    ast.Add: operator.add,

    ast.Sub: operator.sub,

    ast.Mult: operator.mul,

    ast.Div: operator.truediv,

    ast.FloorDiv: operator.floordiv,

    ast.Mod: operator.mod,

    ast.Pow: operator.pow,

    ast.USub: operator.neg,

    ast.UAdd: operator.pos,
}


def calculate_expression(node):

    if isinstance(node, ast.Constant):

        if type(node.value) in (int, float):

            return node.value

        raise ValueError("Invalid number")

    if isinstance(node, ast.BinOp):

        operation = FAST_OPERATORS.get(
            type(node.op)
        )

        if operation is None:

            raise ValueError(
                "Unsupported operation"
            )

        left = calculate_expression(node.left)

        right = calculate_expression(node.right)

        if isinstance(node.op, ast.Pow):

            if abs(right) > 10 or abs(left) > 100000:

                raise ValueError(
                    "Calculation too large"
                )

        result = operation(left, right)

        if (
            not isinstance(result, (int, float))
            or abs(result) > 10**15
        ):

            raise ValueError(
                "Invalid or excessive result"
            )

        return result

    if isinstance(node, ast.UnaryOp):

        operation = FAST_OPERATORS.get(
            type(node.op)
        )

        if operation is None:

            raise ValueError(
                "Invalid operation"
            )

        return operation(
            calculate_expression(node.operand)
        )

    raise ValueError(
        "Invalid expression"
    )


def fast_local_answer(message):

    text = message.lower().strip()

    text = re.sub(
        r"^(what is|calculate|solve|answer|compute)\s+",
        "",
        text
    )

    text = text.rstrip(" ?!.")

    replacements = {

        "multiplied by": "*",

        "multiply by": "*",

        "times": "*",

        "plus": "+",

        "added to": "+",

        "minus": "-",

        "subtracted by": "-",

        "divided by": "/",

        "divide by": "/",

        "over": "/",

        "modulus": "%",

        "mod": "%",
    }

    for word, symbol in replacements.items():

        text = re.sub(
            r"\b" + re.escape(word) + r"\b",
            symbol,
            text
        )

    text = (

        text.replace("×", "*")

        .replace("÷", "/")

        .replace("^", "**")

        .replace(",", "")
    )

    if not re.fullmatch(
        r"[0-9+\-*/().% ]+",
        text
    ):

        return None

    try:

        tree = ast.parse(
            text,
            mode="eval"
        )

        if sum(
            1 for _ in ast.walk(tree)
        ) > 30:

            return None

        answer = calculate_expression(
            tree.body
        )

        if isinstance(answer, float):

            answer = round(answer, 10)

        return str(answer)

    except Exception:

        return None


# ============================================================
# CHAT SYSTEM - SPEED OPTIMIZED
# ============================================================

def normalize_history(history):

    result = []

    for item in history or []:

        if not isinstance(item, dict):

            continue

        role = item.get("role")

        content = item.get("content")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
        ):

            if content.strip():

                result.append({

                    "role": role,

                    "content": content
                })

    return result


# Lower token limits help short responses finish sooner.
# Detailed questions receive more room.

def chat_output_limit(question):

    q = question.lower()

    detailed = any(term in q for term in (

        "explain",

        "teach me",

        "step by step",

        "in detail",

        "chapter",

        "thermodynamics",

        "derive",

        "compare",

        "difference between",

        "why",

        "how does",
    ))

    return 400 if detailed else 150


# Send only four previous messages.
# This reduces the amount of context processed.

def build_chat_contents(history, message):

    contents = []

    for item in history[-4:]:

        role = (
            "model"
            if item["role"] == "assistant"
            else "user"
        )

        contents.append(

            types.Content(

                role=role,

                parts=[
                    types.Part.from_text(
                        text=item["content"]
                    )
                ],
            )
        )

    contents.append(

        types.Content(

            role="user",

            parts=[
                types.Part.from_text(
                    text=message
                )
            ],
        )
    )

    return contents


# ============================================================
# FAST CHAT RESPONSE
# ============================================================

def chat_reply(message, history):

    start_time = time.monotonic()

    history = normalize_history(history)

    if not message or not message.strip():

        yield history

        return

    message = message.strip()

    updated = history + [

        {
            "role": "user",
            "content": message
        }

    ]

    # --------------------------------------------------------
    # 1. LOCAL CALCULATOR
    # --------------------------------------------------------

    local_answer = fast_local_answer(message)

    if local_answer is not None:

        logger.info(
            "CHAT LOCAL CALCULATION"
        )

        yield updated + [

            {
                "role": "assistant",
                "content": local_answer
            }

        ]

        return

    # --------------------------------------------------------
    # 2. CACHE
    # --------------------------------------------------------

    key = cache_key(
        message,
        history
    )

    cached = cache_get(key)

    if cached:

        logger.info(
            "CHAT CACHE HIT"
        )

        yield updated + [

            {
                "role": "assistant",
                "content": cached
            }

        ]

        return

    # --------------------------------------------------------
    # 3. SHOW THINKING STATUS
    # --------------------------------------------------------

    yield updated + [

        {
            "role": "assistant",
            "content": "Thinking..."
        }

    ]

    # --------------------------------------------------------
    # 4. BUILD SHORT CONTEXT
    # --------------------------------------------------------

    contents = build_chat_contents(
        history,
        message
    )

    # --------------------------------------------------------
    # 5. FAST GENERATION SETTINGS
    # --------------------------------------------------------

    config = types.GenerateContentConfig(

        system_instruction=(

            "You are Alpha AI, a fast, accurate, "
            "text-only AI assistant. "

            "Answer the user's actual question directly. "

            "Use clear and simple language. "

            "Keep normal answers short and useful. "

            "For school science and mathematics, "
            "explain important steps and formulas. "

            "For detailed questions, provide a structured "
            "explanation without unnecessary repetition. "

            "Never generate images or videos in Chat."
        ),

        temperature=0.2,

        max_output_tokens=chat_output_limit(message),
    )

    # --------------------------------------------------------
    # 6. PRIMARY MODEL ONLY
    # --------------------------------------------------------

    # Backup model is intentionally disabled in Chat.
    # This avoids waiting for a second model after failure.

    model_name = GEMINI_MODEL

    full_answer = ""

    try:

        logger.info(
            "CHAT START model=%s",
            model_name
        )

        stream = chat_gemini.models.generate_content_stream(

            model=model_name,

            contents=contents,

            config=config,
        )

        # ----------------------------------------------------
        # 7. STREAM RESPONSE
        # ----------------------------------------------------

        for chunk in stream:

            piece = getattr(
                chunk,
                "text",
                None
            )

            if piece:

                full_answer += piece

                yield updated + [

                    {
                        "role": "assistant",
                        "content": full_answer
                    }

                ]

        # ----------------------------------------------------
        # 8. SAVE CACHE
        # ----------------------------------------------------

        if full_answer.strip():

            full_answer = full_answer.strip()

            cache_put(
                key,
                full_answer
            )

            elapsed = (
                time.monotonic() - start_time
            )

            logger.info(
                "CHAT COMPLETE in %.2f seconds",
                elapsed
            )

            return

        raise RuntimeError(
            "Model returned an empty response."
        )

    except Exception as error:

        logger.exception(
            "CHAT ERROR"
        )

        if full_answer.strip():

            cache_put(
                key,
                full_answer.strip()
            )

            return

        logger.error(
            "Chat failed: %s",
            error
        )

        yield updated + [

            {
                "role": "assistant",

                "content": (
                    "Alpha could not connect to its AI model. "
                    "Please try again. If this continues, "
                    "check Render logs and Gemini API status."
                )
            }

        ]


# ============================================================
# GEMINI MEDIA HELPER
# ============================================================

def gemini_generate_media(
    contents,
    config,
    operation
):

    models = [GEMINI_MODEL]

    if (
        GEMINI_BACKUP_MODEL
        and GEMINI_BACKUP_MODEL not in models
    ):

        models.append(
            GEMINI_BACKUP_MODEL
        )

    last_error = None

    for model_name in models:

        try:

            response = gemini.models.generate_content(

                model=model_name,

                contents=contents,

                config=config
            )

            answer = response.text

            if answer and answer.strip():

                return answer.strip()

            raise RuntimeError(
                "Empty model response."
            )

        except Exception as error:

            last_error = error

            logger.exception(
                "%s failed using %s",
                operation,
                model_name
            )

    raise RuntimeError(
        operation + " failed."
    ) from last_error


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path,
    question
):

    if not image_path:

        return "Please upload an image first."

    if not question or not question.strip():

        question = (
            "Describe this image and its important details."
        )

    try:

        with Image.open(image_path) as image:

            image = ImageOps.exif_transpose(
                image
            ).convert("RGB")

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=90
            )

            image_bytes = buffer.getvalue()

        contents = [

            question,

            types.Part.from_bytes(

                data=image_bytes,

                mime_type="image/jpeg"
            )
        ]

        config = types.GenerateContentConfig(

            max_output_tokens=1200,

            temperature=0.3
        )

        return gemini_generate_media(

            contents,

            config,

            "IMAGE ANALYSIS"
        )

    except Exception:

        logger.exception(
            "IMAGE ANALYSIS ERROR"
        )

        return (
            "Image analysis failed. Please try again."
        )


# ============================================================
# IMAGE VARIATION
# ============================================================

def generate_from_image_analysis(
    image_path,
    description,
    request: gr.Request
):

    if not image_path or not description:

        return None, None

    analysis = analyze_image(

        image_path,

        "Describe this image for creating a new image. "
        "User request: " + description
    )

    if "analysis failed" in analysis.lower():

        return None, None

    prompt = (

        "Create a new image based on this visual description:\n"

        + description

        + "\n"

        + analysis
    )

    try:

        path = create_image(

            prompt,

            request,

            ratio="16:9"
        )

        return path, path

    except Exception:

        logger.exception(
            "IMAGE VARIATION ERROR"
        )

        return None, None


# ============================================================
# VIDEO HELPERS
# ============================================================

def get_video_path(video):

    if isinstance(video, str):

        return video

    if isinstance(video, dict):

        return (
            video.get("video")
            or video.get("path")
        )

    return None


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(
    video,
    question
):

    video_path = get_video_path(video)

    if not video_path:

        return "Please upload a video first."

    uploaded = None

    try:

        if not os.path.isfile(video_path):

            return "Video file not found."

        file_size = os.path.getsize(
            video_path
        )

        if file_size > VIDEO_MAX_SIZE:

            return (
                "Video must be smaller than 50 MB."
            )

        if file_size == 0:

            return "The video file is empty."

        uploaded = gemini.files.upload(
            file=video_path
        )

        for _ in range(90):

            uploaded = gemini.files.get(
                name=uploaded.name
            )

            state = getattr(
                uploaded,
                "state",
                None
            )

            state_name = getattr(
                state,
                "name",
                str(state)
            ).upper()

            if state_name == "ACTIVE":

                break

            if state_name == "FAILED":

                raise RuntimeError(
                    "Video processing failed."
                )

            time.sleep(2)

        else:

            raise TimeoutError(
                "Video processing timed out."
            )

        if not question or not question.strip():

            question = (
                "Summarize this video and explain "
                "its important scenes."
            )

        config = types.GenerateContentConfig(

            max_output_tokens=1500,

            temperature=0.3
        )

        return gemini_generate_media(

            [uploaded, question],

            config,

            "VIDEO ANALYSIS"
        )

    except Exception:

        logger.exception(
            "VIDEO ANALYSIS ERROR"
        )

        return (
            "Video analysis failed. Please try again."
        )

    finally:

        if uploaded is not None:

            try:

                gemini.files.delete(
                    name=uploaded.name
                )

            except Exception:

                logger.exception(
                    "VIDEO FILE CLEANUP ERROR"
                )


# ============================================================
# VIDEO IMAGE GENERATION
# ============================================================

def generate_from_video(
    video,
    description,
    request: gr.Request
):

    if not get_video_path(video):

        return None, None

    analysis = analyze_video(

        video,

        "Describe the important scenes for image creation. "
        + (description or "")
    )

    if "failed" in analysis.lower():

        return None, None

    prompt = (

        "Create a new still image inspired by this video analysis:\n"

        + (description or "")

        + "\n"

        + analysis
    )

    try:

        path = create_image(

            prompt,

            request,

            ratio="16:9"
        )

        return path, path

    except Exception:

        logger.exception(
            "VIDEO IMAGE GENERATION ERROR"
        )

        return None, None


# ============================================================
# IMAGE STUDIO CALLBACK
# ============================================================

def make_image(
    prompt,
    style,
    extra,
    ratio,
    request: gr.Request
):

    try:

        path = create_image(

            prompt,

            request,

            ratio,

            style,

            extra
        )

        return path, path

    except Exception:

        logger.exception(
            "IMAGE STUDIO ERROR"
        )

        return None, None


# ============================================================
# CSS
# ============================================================

CSS = """

.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#header {
    text-align: center;
    padding: 25px 10px;
}

#header h1 {
    font-size: 46px;
    font-weight: 900;
    letter-spacing: 4px;
}

#header p {
    font-size: 16px;
    letter-spacing: 2px;
}

button, textarea {
    border-radius: 12px !important;
}

#result {
    border-radius: 16px;
    overflow: hidden;
}

"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="ALPHA AI - Think Beyond Limits"
) as app:

    gr.HTML("""

    <div id="header">

        <h1>🚀 ALPHA AI 16.0</h1>

        <p>THINK BEYOND LIMITS</p>

        <p>YOUR AI ASSISTANT</p>

    </div>

    """)

    with gr.Tabs():

        # ====================================================
        # CHAT
        # ====================================================

        with gr.Tab("💬 Chat"):

            chatbot = gr.Chatbot(

                label="Alpha AI",

                height=500
            )

            chat_input = gr.Textbox(

                placeholder="Ask Alpha anything...",

                label="Message",

                lines=2
            )

            with gr.Row():

                chat_send = gr.Button(

                    "Send",

                    variant="primary"
                )

                chat_clear = gr.Button(
                    "Clear Chat"
                )

            chat_send.click(

                fn=chat_reply,

                inputs=[
                    chat_input,
                    chatbot
                ],

                outputs=chatbot,

                api_name="chat",

            ).then(

                fn=lambda: "",

                outputs=chat_input
            )

            chat_input.submit(

                fn=chat_reply,

                inputs=[
                    chat_input,
                    chatbot
                ],

                outputs=chatbot,

                api_name="chat_submit",

            ).then(

                fn=lambda: "",

                outputs=chat_input
            )

            chat_clear.click(

                fn=lambda: ([], ""),

                outputs=[
                    chatbot,
                    chat_input
                ]
            )

        # ====================================================
        # IMAGE STUDIO
        # ====================================================

        with gr.Tab("🎨 Image Studio"):

            image_prompt = gr.Textbox(

                label="Image Description",

                placeholder="Describe the image you want...",

                lines=4
            )

            with gr.Row():

                image_style = gr.Dropdown(

                    choices=list(STYLES.keys()),

                    value="Automatic",

                    label="Style"
                )

                image_ratio = gr.Dropdown(

                    choices=list(RATIOS.keys()),

                    value="16:9",

                    label="Aspect Ratio"
                )

            image_extra = gr.Textbox(

                label="Additional Instructions",

                placeholder="Optional...",

                lines=2
            )

            image_generate = gr.Button(

                "🚀 Generate Image",

                variant="primary"
            )

            studio_output = gr.Image(

                label="Generated Image",

                type="filepath",

                interactive=False,

                elem_id="result"
            )

            studio_download = gr.File(
                label="Download Image"
            )

            image_generate.click(

                fn=make_image,

                inputs=[

                    image_prompt,

                    image_style,

                    image_extra,

                    image_ratio
                ],

                outputs=[

                    studio_output,

                    studio_download
                ],

                api_name="generate_image"
            )

        # ====================================================
        # IMAGE ANALYSIS
        # ====================================================

        with gr.Tab("🖼️ Image Analysis"):

            analysis_image = gr.Image(

                label="Upload Image",

                type="filepath"
            )

            image_question = gr.Textbox(

                label="Ask About Your Image",

                placeholder="What is in this image?",

                lines=2
            )

            analyze_image_button = gr.Button(

                "🔍 Analyze Image",

                variant="primary"
            )

            image_analysis_output = gr.Textbox(

                label="Analysis",

                lines=8,

                interactive=False
            )

            analyze_image_button.click(

                fn=analyze_image,

                inputs=[

                    analysis_image,

                    image_question
                ],

                outputs=image_analysis_output,

                api_name="analyze_image"
            )

            gr.Markdown(
                "### Generate a New Image Inspired by Your Upload"
            )

            image_variation_prompt = gr.Textbox(

                label="Describe the New Image",

                placeholder="Describe what you want to create...",

                lines=2
            )

            image_variation_button = gr.Button(

                "✨ Generate Image",

                variant="primary"
            )

            image_variation_output = gr.Image(

                label="Generated Image",

                type="filepath",

                interactive=False
            )

            image_variation_download = gr.File(
                label="Download Image"
            )

            image_variation_button.click(

                fn=generate_from_image_analysis,

                inputs=[

                    analysis_image,

                    image_variation_prompt
                ],

                outputs=[

                    image_variation_output,

                    image_variation_download
                ],

                api_name="generate_from_image"
            )

        # ====================================================
        # VIDEO ANALYSIS
        # ====================================================

        with gr.Tab("🎬 Video Analysis"):

            video_input = gr.Video(

                label="Upload Video",

                sources=["upload"]
            )

            video_question = gr.Textbox(

                label="Ask About Your Video",

                placeholder="Summarize this video...",

                lines=2
            )

            video_analyze_button = gr.Button(

                "🔍 Analyze Video",

                variant="primary"
            )

            video_analysis_output = gr.Textbox(

                label="Video Analysis",

                lines=9,

                interactive=False
            )

            video_analyze_button.click(

                fn=analyze_video,

                inputs=[

                    video_input,

                    video_question
                ],

                outputs=video_analysis_output,

                api_name="analyze_video"
            )

            gr.Markdown(
                "### Generate an Image From Video Analysis"
            )

            video_image_prompt = gr.Textbox(

                label="Image Description",

                placeholder="Describe the scene you want to create...",

                lines=2
            )

            video_generate_button = gr.Button(

                "✨ Generate Image",

                variant="primary"
            )

            video_image_output = gr.Image(

                label="Generated Image",

                type="filepath",

                interactive=False
            )

            video_download = gr.File(
                label="Download Image"
            )

            video_generate_button.click(

                fn=generate_from_video,

                inputs=[

                    video_input,

                    video_image_prompt
                ],

                outputs=[

                    video_image_output,

                    video_download
                ],

                api_name="generate_from_video"
            )


# ============================================================
# QUEUE
# ============================================================

app.queue(

    max_size=40,

    default_concurrency_limit=4
)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    logger.info(
        "ALPHA AI 16.0 STARTING"
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        show_error=False,

        theme=gr.themes.Soft(

            primary_hue="blue",

            secondary_hue="slate"
        ),

        css=CSS
)
