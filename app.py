import os
import io
import re
import ast
import uuid
import time
import base64
import hashlib
import logging
import operator
import threading
import subprocess

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import OrderedDict

import gradio as gr
import requests

from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI
# THINK BEYOND LIMITS
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALPHA")


# ============================================================
# CONFIGURATION
# ============================================================

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

QUOTA_SALT = os.environ.get("ALPHA_QUOTA_SALT")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

GEMINI_BACKUP_MODEL = os.environ.get(
    "GEMINI_BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)

PORT = int(os.environ.get("PORT", "10000"))

CHAT_TIMEOUT = 30000
API_TIMEOUT = 120000
IMAGE_TIMEOUT = 120

VIDEO_MAX_SIZE = 50 * 1024 * 1024
AUDIO_MAX_SIZE = 25 * 1024 * 1024

IST = ZoneInfo("Asia/Kolkata")

BASE_DIR = Path("/tmp/alpha")
IMAGE_DIR = BASE_DIR / "images"
VIDEO_DIR = BASE_DIR / "videos"
TRANSLATION_DIR = BASE_DIR / "translations"

for directory in (
    IMAGE_DIR,
    VIDEO_DIR,
    TRANSLATION_DIR
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# FREE PLAN LIMITS
# ============================================================

FREE_LIMITS = {
    "audio_translate": 3,
    "video_translate": 3,
    "video_analysis": 5,
    "image_analysis": 5,
    "image_upload": 10,
    "image_create": 10,
    "video_create": 1,
}


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

chat_gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=CHAT_TIMEOUT
    )
)

gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=API_TIMEOUT
    )
)

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    + CLOUDFLARE_ACCOUNT_ID
    + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
)


# ============================================================
# MATHEMATICAL FORMATTING
# ============================================================

LATEX_REPLACEMENTS = {
    r"\Delta": "Δ",
    r"\delta": "δ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\omega": "ω",
    r"\Omega": "Ω",
    r"\Sigma": "Σ",
    r"\Phi": "Φ",
    r"\phi": "φ",
    r"\times": "×",
    r"\div": "÷",
    r"\cdot": "·",
    r"\pm": "±",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\equiv": "≡",
    r"\infty": "∞",
    r"\degree": "°",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\Rightarrow": "⇒",
    r"\sum": "Σ",
    r"\sqrt": "√",
    r"\propto": "∝",
    r"\therefore": "∴",
    r"\because": "∵",
    r"\partial": "∂",
    r"\nabla": "∇",
    r"\int": "∫",
}


def format_math_text(text):

    if not isinstance(text, str):
        return text

    for latex, symbol in LATEX_REPLACEMENTS.items():
        text = text.replace(latex, symbol)

    text = text.replace(r"\(", "")
    text = text.replace(r"\)", "")
    text = text.replace(r"\[", "")
    text = text.replace(r"\]", "")

    text = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"(\1)/(\2)",
        text
    )

    text = re.sub(
        r"\\sqrt\{([^{}]+)\}",
        r"√(\1)",
        text
    )

    text = text.replace(r"^{2}", "²")
    text = text.replace(r"^{3}", "³")

    text = text.replace(r"_{1}", "₁")
    text = text.replace(r"_{2}", "₂")
    text = text.replace(r"_{3}", "₃")

    text = text.replace(r"\%", "%")
    text = text.replace(r"\_", "_")

    return text


# ============================================================
# CACHE
# ============================================================

CACHE_TTL = 1800
CACHE_MAX = 128

_answer_cache = OrderedDict()
_cache_lock = threading.Lock()


def cache_get(key):

    with _cache_lock:

        item = _answer_cache.get(key)

        if not item:
            return None

        created, value = item

        if time.monotonic() - created > CACHE_TTL:
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

        while len(_answer_cache) > CACHE_MAX:
            _answer_cache.popitem(last=False)


def make_cache_key(message, history):

    raw = repr(history[-4:]) + message

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
        logger.exception("Visitor identification failed")

    return hashlib.sha256(
        (QUOTA_SALT + ":" + ip).encode()
    ).hexdigest()


def supabase_headers():

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": "Bearer " + SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": "application/json",
    }


# ============================================================
# INDEPENDENT QUOTA SYSTEM
# ============================================================

def consume_quota(visitor, feature):

    limit = FREE_LIMITS[feature]

    endpoint = (
        SUPABASE_URL
        + "/rest/v1/rpc/consume_alpha_media_quota"
    )

    payload = {
        "p_visitor_hash": visitor,
        "p_usage_date": today(),
        "p_media_type": feature,
        "p_daily_limit": limit,
    }

    response = http.post(
        endpoint,
        headers=supabase_headers(),
        json=payload,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError("Invalid quota response.")

    if "allowed" not in result:
        raise RuntimeError("Quota response missing allowed field.")

    return result


def refund_quota(visitor, usage_date, feature):

    endpoint = (
        SUPABASE_URL
        + "/rest/v1/rpc/refund_alpha_media_quota"
    )

    payload = {
        "p_visitor_hash": visitor,
        "p_usage_date": usage_date,
        "p_media_type": feature,
    }

    try:

        response = http.post(
            endpoint,
            headers=supabase_headers(),
            json=payload,
            timeout=20
        )

        response.raise_for_status()

    except Exception:

        logger.exception(
            "QUOTA REFUND FAILED: %s",
            feature
        )


def reserve_quota(visitor, feature):

    result = consume_quota(
        visitor,
        feature
    )

    if not result.get("allowed"):

        raise RuntimeError(
            "Daily limit reached for "
            + feature.replace("_", " ")
            + ". Your Free Plan limit is "
            + str(FREE_LIMITS[feature])
            + " per day."
        )

    return today()


# ============================================================
# GEMINI HELPER
# ============================================================

def gemini_generate(contents, config, operation):

    models = [GEMINI_MODEL]

    if GEMINI_BACKUP_MODEL not in models:
        models.append(GEMINI_BACKUP_MODEL)

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
                return format_math_text(answer.strip())

            raise RuntimeError("Empty model response.")

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
# IMAGE STUDIO
# ============================================================

STYLES = {
    "Automatic": "",
    "Photorealistic": "Photorealistic, realistic lighting and details.",
    "Cinematic": "Cinematic composition and dramatic lighting.",
    "Anime": "High quality anime illustration.",
    "3D Render": "Professional detailed 3D render.",
    "Digital Art": "Professional digital artwork.",
    "Watercolor": "Traditional watercolor painting.",
    "Oil Painting": "Detailed oil painting.",
    "Pixel Art": "High quality pixel art.",
    "Minimalist": "Clean minimalist composition.",
    "Product Photography": "Professional studio product photography.",
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


def generate_image_bytes(prompt):

    response = http.post(
        CLOUDFLARE_URL,
        headers={
            "Authorization": "Bearer " + CLOUDFLARE_API_TOKEN,
            "Content-Type": "application/json",
        },
        json={
            "prompt": prompt[:2048],
            "steps": 4,
        },
        timeout=(10, IMAGE_TIMEOUT)
    )

    response.raise_for_status()

    if "application/json" in response.headers.get(
        "Content-Type", ""
    ).lower():

        data = response.json()

        if not data.get("success"):
            raise RuntimeError("Image generation failed.")

        encoded = data.get("result", {}).get("image")

        if not encoded:
            raise RuntimeError("No image returned.")

        image_bytes = base64.b64decode(encoded)

    else:

        image_bytes = response.content

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()

    return image_bytes


def export_image(image_bytes, ratio="16:9"):

    width, height = RATIOS.get(
        ratio,
        (1280, 720)
    )

    with Image.open(io.BytesIO(image_bytes)) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        result = ImageOps.pad(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            color=(15, 15, 20)
        )

        output = IMAGE_DIR / (uuid.uuid4().hex + ".jpg")

        result.save(
            output,
            format="JPEG",
            quality=94
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
        raise ValueError("Enter an image description.")

    if request is None:
        raise RuntimeError("Visitor information unavailable.")

    visitor = get_visitor(request)
    usage_date = reserve_quota(visitor, "image_create")

    try:

        final_prompt = (
            prompt.strip()
            + "\n"
            + STYLES.get(style, "")
            + "\n"
            + (extra or "")
        )

        image_bytes = generate_image_bytes(
            final_prompt
        )

        return export_image(
            image_bytes,
            ratio
        )

    except Exception:

        refund_quota(
            visitor,
            usage_date,
            "image_create"
        )

        raise


def make_image(
    prompt,
    style,
    extra,
    ratio,
    request: gr.Request = None
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

    except Exception as error:

        logger.exception("IMAGE STUDIO ERROR")

        raise gr.Error(str(error))


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

        operation = FAST_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operation")

        left = calculate_expression(node.left)
        right = calculate_expression(node.right)

        if isinstance(node.op, ast.Pow):

            if abs(right) > 10 or abs(left) > 100000:
                raise ValueError("Calculation too large")

        result = operation(left, right)

        if (
            not isinstance(result, (int, float))
            or abs(result) > 10**15
        ):
            raise ValueError("Invalid result")

        return result

    if isinstance(node, ast.UnaryOp):

        operation = FAST_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Invalid operation")

        return operation(
            calculate_expression(node.operand)
        )

    raise ValueError("Invalid expression")


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

    if not re.fullmatch(r"[0-9+\-*/().% ]+", text):
        return None

    try:

        tree = ast.parse(text, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > 30:
            return None

        answer = calculate_expression(tree.body)

        if isinstance(answer, float):
            answer = round(answer, 10)

        return str(answer)

    except Exception:
        return None


# ============================================================
# CHAT
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
            and content.strip()
        ):

            result.append({
                "role": role,
                "content": content
            })

    return result


def chat_output_limit(question):

    detailed = any(
        term in question.lower()
        for term in (
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
        )
    )

    return 850 if detailed else 350


def chat_reply(message, history):

    history = normalize_history(history)

    if not message or not message.strip():
        yield history
        return

    message = message.strip()

    updated = history + [{
        "role": "user",
        "content": message
    }]

    local_answer = fast_local_answer(message)

    if local_answer is not None:

        yield updated + [{
            "role": "assistant",
            "content": local_answer
        }]

        return

    key = make_cache_key(message, history)
    cached = cache_get(key)

    if cached:

        yield updated + [{
            "role": "assistant",
            "content": cached
        }]

        return

    yield updated + [{
        "role": "assistant",
        "content": "Thinking..."
    }]

    contents = []

    for item in history[-8:]:

        contents.append(
            types.Content(
                role=(
                    "model"
                    if item["role"] == "assistant"
                    else "user"
                ),
                parts=[
                    types.Part.from_text(
                        text=item["content"]
                    )
                ]
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=message)
            ]
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are Alpha AI, a fast and accurate assistant. "
            "Answer directly using clear and simple language. "
            "For school science and mathematics, explain logically "
            "and include useful formulas and examples. "
            "Use Unicode mathematical symbols instead of raw LaTeX. "
            "Do not use unnecessary verbosity. "
            "Chat is text-only. Never generate images or videos in Chat."
        ),
        temperature=0.2,
        max_output_tokens=chat_output_limit(message)
    )

    full_answer = ""
    last_error = None

    for model_name in (
        GEMINI_MODEL,
        GEMINI_BACKUP_MODEL
    ):

        if not model_name:
            continue

        try:

            stream = chat_gemini.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            )

            for chunk in stream:

                piece = getattr(chunk, "text", None)

                if piece:

                    full_answer += piece

                    yield updated + [{
                        "role": "assistant",
                        "content": format_math_text(full_answer)
                    }]

            if full_answer.strip():

                full_answer = format_math_text(
                    full_answer.strip()
                )

                cache_put(key, full_answer)

                return

            raise RuntimeError("Empty model response.")

        except Exception as error:

            last_error = error

            logger.exception(
                "CHAT FAILED: %s",
                model_name
            )

            if full_answer:
                break

    if full_answer.strip():

        full_answer = format_math_text(
            full_answer.strip()
        )

        cache_put(key, full_answer)

        return

    logger.error("CHAT ERROR: %s", last_error)

    yield updated + [{
        "role": "assistant",
        "content": (
            "Alpha could not connect to its AI model. "
            "Please try again."
        )
    }]


# ============================================================
# IMAGE UPLOAD AND IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path,
    question,
    request: gr.Request = None
):

    if not image_path:
        return "Please upload an image first."

    if request is None:
        return "Visitor information unavailable."

    visitor = get_visitor(request)
    usage_date = today()

    upload_reserved = False
    analysis_reserved = False

    try:

        reserve_quota(visitor, "image_upload")
        upload_reserved = True

        reserve_quota(visitor, "image_analysis")
        analysis_reserved = True

        if not question or not question.strip():

            question = (
                "Describe this image and its important details."
            )

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

        return gemini_generate(
            contents,
            config,
            "IMAGE ANALYSIS"
        )

    except Exception as error:

        if upload_reserved:
            refund_quota(
                visitor,
                usage_date,
                "image_upload"
            )

        if analysis_reserved:
            refund_quota(
                visitor,
                usage_date,
                "image_analysis"
            )

        logger.exception("IMAGE ANALYSIS ERROR")

        return "Image analysis failed: " + str(error)


def generate_from_image_analysis(
    image_path,
    description,
    request: gr.Request = None
):

    if not image_path or not description:
        raise gr.Error("Upload an image and enter a description.")

    with Image.open(image_path) as image:

        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        buffer = io.BytesIO()

        image.save(buffer, format="JPEG", quality=90)

        image_bytes = buffer.getvalue()

    analysis = gemini_generate(
        [
            (
                "Describe this image for creating a new image. "
                "User request: " + description
            ),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        types.GenerateContentConfig(
            max_output_tokens=800,
            temperature=0.3
        ),
        "IMAGE DESCRIPTION"
    )

    path = create_image(
        description + "\n" + analysis,
        request,
        ratio="16:9"
    )

    return path, path


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


def wait_for_gemini_file(uploaded, timeout_seconds=180):

    deadline = time.monotonic() + timeout_seconds

    current = uploaded

    while time.monotonic() < deadline:

        current = gemini.files.get(
            name=current.name
        )

        state = getattr(current, "state", None)

        state_name = getattr(
            state,
            "name",
            str(state)
        ).upper()

        if state_name.endswith("ACTIVE"):
            return current

        if state_name.endswith("FAILED"):
            raise RuntimeError("Media processing failed.")

        time.sleep(2)

    raise TimeoutError("Media processing timed out.")


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(
    video,
    question,
    request: gr.Request = None
):

    video_path = get_video_path(video)

    if not video_path:
        return "Please upload a video first."

    if not os.path.isfile(video_path):
        return "Video file not found."

    if os.path.getsize(video_path) > VIDEO_MAX_SIZE:
        return "Video must be smaller than 50 MB."

    if os.path.getsize(video_path) == 0:
        return "Video file is empty."

    if request is None:
        return "Visitor information unavailable."

    visitor = get_visitor(request)
    usage_date = today()

    reserved = False
    uploaded = None

    try:

        reserve_quota(visitor, "video_analysis")
        reserved = True

        uploaded = gemini.files.upload(
            file=video_path
        )

        uploaded = wait_for_gemini_file(uploaded)

        if not question or not question.strip():

            question = (
                "Summarize this video and explain its important scenes."
            )

        result = gemini_generate(
            [uploaded, question],
            types.GenerateContentConfig(
                max_output_tokens=1500,
                temperature=0.3
            ),
            "VIDEO ANALYSIS"
        )

        return result

    except Exception as error:

        if reserved:
            refund_quota(
                visitor,
                usage_date,
                "video_analysis"
            )

        logger.exception("VIDEO ANALYSIS ERROR")

        return "Video analysis failed: " + str(error)

    finally:

        if uploaded is not None:

            try:
                gemini.files.delete(name=uploaded.name)

            except Exception:
                logger.exception("VIDEO CLEANUP ERROR")


# ============================================================
# AUDIO AND VIDEO TRANSLATION
# ============================================================

TRANSLATION_LANGUAGES = [
    "Hindi",
    "English",
    "Gujarati",
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Chinese",
    "Arabic",
    "Portuguese",
    "Russian",
    "Korean",
    "Italian",
    "Bengali",
    "Tamil",
    "Telugu",
    "Marathi",
    "Urdu",
]

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a",
    ".aac", ".ogg", ".flac"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".webm",
    ".avi", ".mpeg", ".mpg"
}


def translate_media_file(
    media_path,
    target_language,
    media_kind,
    request=None
):

    if (
        not media_path
        or not os.path.isfile(media_path)
    ):
        raise ValueError("Please upload a media file.")

    extension = Path(media_path).suffix.lower()

    visitor = None
    usage_date = today()
    reserved = False

    if media_kind == "audio":

        if extension not in AUDIO_EXTENSIONS:
            raise ValueError("Unsupported audio format.")

        if os.path.getsize(media_path) > AUDIO_MAX_SIZE:
            raise ValueError("Audio must be 25 MB or smaller.")

        feature = "audio_translate"

    else:

        if extension not in VIDEO_EXTENSIONS:
            raise ValueError("Unsupported video format.")

        if os.path.getsize(media_path) > VIDEO_MAX_SIZE:
            raise ValueError("Video must be 50 MB or smaller.")

        if request is None:
            raise RuntimeError("Visitor information unavailable.")

        feature = "video_translate"

    uploaded = None

    try:

        if media_kind == "video":

            visitor = get_visitor(request)

            reserve_quota(
                visitor,
                feature
            )

            reserved = True

        uploaded = gemini.files.upload(
            file=media_path
        )

        uploaded = wait_for_gemini_file(uploaded)

        if media_kind == "audio":

            prompt = (
                "Translate the spoken content in this audio into "
                + target_language
                + ". Identify the original language. "
                "Provide a faithful and complete translated transcript. "
                "Do not summarize or invent words. "
                "Mark unclear sections as [unclear]."
            )

            output_name = uuid.uuid4().hex + ".txt"

        else:

            prompt = (
                "Translate all spoken dialogue in this video into "
                + target_language
                + ". Return ONLY valid SRT subtitle text. "
                "Include sequential subtitle numbers and timestamps "
                "in HH:MM:SS,mmm --> HH:MM:SS,mmm format. "
                "Do not invent dialogue."
            )

            output_name = uuid.uuid4().hex + ".srt"

        result = gemini_generate(
            [uploaded, prompt],
            types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192
            ),
            media_kind.upper() + " TRANSLATION"
        )

        result = result.strip()

        if media_kind == "video":

            result = re.sub(
                r"^```(?:srt|text)?\s*|\s*```$",
                "",
                result,
                flags=re.IGNORECASE
            ).strip()

            if not re.search(
                r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->",
                result
            ):
                raise RuntimeError(
                    "Invalid SRT output. Please try another video."
                )

        output_path = TRANSLATION_DIR / output_name

        output_path.write_text(
            result,
            encoding="utf-8"
        )

        return result, str(output_path)

    except Exception:

        if reserved and visitor:

            refund_quota(
                visitor,
                usage_date,
                feature
            )

        raise

    finally:

        if uploaded is not None:

            try:
                gemini.files.delete(name=uploaded.name)

            except Exception:
                logger.exception("TRANSLATION CLEANUP ERROR")


def translate_audio(
    audio_path,
    target_language,
    request: gr.Request = None
):

    if request is None:
        raise gr.Error("Visitor information unavailable.")

    visitor = get_visitor(request)
    usage_date = today()

    reserved = False

    try:

        reserve_quota(
            visitor,
            "audio_translate"
        )

        reserved = True

        return translate_media_file(
            audio_path,
            target_language,
            "audio"
        )

    except Exception as error:

        if reserved:
            refund_quota(
                visitor,
                usage_date,
                "audio_translate"
            )

        logger.exception("AUDIO TRANSLATION ERROR")

        raise gr.Error(str(error))


def translate_video(
    video,
    target_language,
    request: gr.Request = None
):

    try:

        return translate_media_file(
            get_video_path(video),
            target_language,
            "video",
            request
        )

    except Exception as error:

        logger.exception("VIDEO TRANSLATION ERROR")

        raise gr.Error(str(error))


# ============================================================
# VIDEO CREATION
# ============================================================

def create_video(
    topic,
    duration,
    ratio,
    request: gr.Request = None
):

    if not topic or not topic.strip():
        raise gr.Error("Enter a video topic.")

    if request is None:
        raise gr.Error("Visitor information unavailable.")

    visitor = get_visitor(request)
    usage_date = today()

    reserved = False

    try:

        reserve_quota(
            visitor,
            "video_create"
        )

        reserved = True

        script = gemini_generate(
            (
                "Create a visual storyboard for this video topic: "
                + topic
                + ". Return exactly 3 numbered scenes. "
                "Each scene must have a short visual description. "
                "Do not include dangerous or misleading real-world claims. "
                "Label fictional news scenes as illustrative."
            ),
            types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=900
            ),
            "VIDEO STORYBOARD"
        )

        scene_prompts = gemini_generate(
            (
                "Convert this storyboard into exactly three image prompts. "
                "Return only three prompts, one per line, numbered 1, 2, 3. "
                "Make all scenes visually consistent.\n"
                + script
            ),
            types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=700
            ),
            "VIDEO SCENE PROMPTS"
        )

        prompts = re.findall(
            r"^\s*(?:[1-3][.)-]\s*)(.+)$",
            scene_prompts,
            flags=re.MULTILINE
        )

        if len(prompts) < 3:

            raise RuntimeError(
                "Could not create three video scenes."
            )

        scene_files = []

        for scene_prompt in prompts[:3]:

            image_bytes = generate_image_bytes(
                scene_prompt
            )

            scene_path = export_image(
                image_bytes,
                ratio
            )

            scene_files.append(scene_path)

        ffmpeg = os.environ.get(
            "FFMPEG_PATH",
            "ffmpeg"
        )

        output_path = (
            VIDEO_DIR
            / (uuid.uuid4().hex + ".mp4")
        )

        width, height = RATIOS.get(
            ratio,
            (1280, 720)
        )

        seconds = max(
            3,
            min(int(duration), 30)
        )

        scene_duration = seconds / 3

        command = [
            ffmpeg,
            "-y",
        ]

        for scene in scene_files:

            command.extend([
                "-loop", "1",
                "-t", str(scene_duration),
                "-i", scene
            ])

        filter_complex = (
            "[0:v]scale="
            + str(width) + ":" + str(height)
            + ":force_original_aspect_ratio=decrease,"
            "pad="
            + str(width) + ":" + str(height)
            + ":(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
            "[1:v]scale="
            + str(width) + ":" + str(height)
            + ":force_original_aspect_ratio=decrease,"
            "pad="
            + str(width) + ":" + str(height)
            + ":(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
            "[2:v]scale="
            + str(width) + ":" + str(height)
            + ":force_original_aspect_ratio=decrease,"
            "pad="
            + str(width) + ":" + str(height)
            + ":(ow-iw)/2:(oh-ih)/2,setsar=1[v2];"
            "[v0][v1][v2]concat=n=3:v=1:a=0[outv]"
        )

        command.extend([
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path)
        ])

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180
        )

        if process.returncode != 0:
            raise RuntimeError(
                "FFmpeg video rendering failed: "
                + process.stderr[-1000:]
            )

        if not output_path.exists():
            raise RuntimeError("Video output was not created.")

        return str(output_path), str(output_path)

    except Exception:

        if reserved:

            refund_quota(
                visitor,
                usage_date,
                "video_create"
            )

        logger.exception("VIDEO CREATION ERROR")

        raise gr.Error(
            "Video creation failed. Check Render logs and FFmpeg."
        )


# ============================================================
# IMAGE VARIATION FROM VIDEO
# ============================================================

def make_image_from_video(
    video,
    description,
    request: gr.Request = None
):

    video_path = get_video_path(video)

    if not video_path:
        raise gr.Error("Upload a video first.")

    analysis = analyze_video(
        video,
        "Describe the important scenes in this video.",
        request
    )

    if "failed" in analysis.lower():
        raise gr.Error(analysis)

    path = create_image(
        (description or "Create an image inspired by this video.")
        + "\n"
        + analysis,
        request,
        ratio="16:9"
    )

    return path, path


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
        <h1>🚀 ALPHA AI</h1>
        <p>THINK BEYOND LIMITS</p>
        <p>YOUR AI ASSISTANT</p>
    </div>
    """)

    with gr.Tabs():

        # ----------------------------------------------------
        # CHAT
        # ----------------------------------------------------

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

                chat_clear = gr.Button("Clear Chat")

            chat_send.click(
                fn=chat_reply,
                inputs=[chat_input, chatbot],
                outputs=chatbot,
                api_name="chat"
            ).then(
                fn=lambda: "",
                outputs=chat_input
            )

            chat_input.submit(
                fn=chat_reply,
                inputs=[chat_input, chatbot],
                outputs=chatbot,
                api_name="chat_submit"
            ).then(
                fn=lambda: "",
                outputs=chat_input
            )

            chat_clear.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, chat_input]
            )

        # ----------------------------------------------------
        # IMAGE STUDIO
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IMAGE ANALYSIS AND UPLOAD
        # ----------------------------------------------------

        with gr.Tab("🖼️ Image Analysis"):

            gr.Markdown(
                "Upload an image and ask Alpha about it."
            )

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

            gr.Markdown("### Create an Image Inspired by Your Upload")

            image_variation_prompt = gr.Textbox(
                label="New Image Description",
                lines=2
            )

            image_variation_button = gr.Button(
                "✨ Generate Image",
                variant="primary"
            )

            image_variation_output = gr.Image(
                label="Generated Image",
                type="filepath"
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

        # ----------------------------------------------------
        # AUDIO TRANSLATOR
        # ----------------------------------------------------

        with gr.Tab("🎧 Audio Translator"):

            gr.Markdown(
                "Translate spoken audio into another language. "
                "Output is translated text, not dubbed audio."
            )

            audio_input = gr.Audio(
                label="Upload Audio",
                sources=["upload"],
                type="filepath"
            )

            audio_language = gr.Dropdown(
                choices=TRANSLATION_LANGUAGES,
                value="Hindi",
                label="Translate To"
            )

            audio_translate_button = gr.Button(
                "🌐 Translate Audio",
                variant="primary"
            )

            audio_translation_output = gr.Textbox(
                label="Translated Transcript",
                lines=12,
                interactive=False
            )

            audio_translation_download = gr.File(
                label="Download Transcript"
            )

            audio_translate_button.click(
                fn=translate_audio,
                inputs=[
                    audio_input,
                    audio_language
                ],
                outputs=[
                    audio_translation_output,
                    audio_translation_download
                ],
                api_name="translate_audio"
            )

        # ----------------------------------------------------
        # VIDEO TRANSLATOR
        # ----------------------------------------------------

        with gr.Tab("🎞️ Video Translator"):

            gr.Markdown(
                "Create translated subtitles (SRT) from a video. "
                "This does not create dubbed speech."
            )

            translation_video_input = gr.Video(
                label="Upload Video",
                sources=["upload"]
            )

            video_language = gr.Dropdown(
                choices=TRANSLATION_LANGUAGES,
                value="Hindi",
                label="Translate To"
            )

            video_translate_button = gr.Button(
                "🌐 Translate Video",
                variant="primary"
            )

            video_translation_output = gr.Textbox(
                label="Translated Subtitles",
                lines=12,
                interactive=False
            )

            video_translation_download = gr.File(
                label="Download SRT"
            )

            video_translate_button.click(
                fn=translate_video,
                inputs=[
                    translation_video_input,
                    video_language
                ],
                outputs=[
                    video_translation_output,
                    video_translation_download
                ],
                api_name="translate_video"
            )

        # ----------------------------------------------------
        # VIDEO ANALYSIS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # VIDEO CREATION
        # ----------------------------------------------------

        with gr.Tab("🎥 Video Studio"):

            gr.Markdown(
                """
                ### Alpha AI Video Studio

                Create a short video using AI-generated scenes.

                This version creates a slideshow-style video from
                generated images. It does not generate moving AI
                footage or a spoken voiceover.
                """
            )

            video_topic = gr.Textbox(
                label="Video Topic",
                placeholder="Example: India's space programme",
                lines=3
            )

            with gr.Row():

                video_duration = gr.Dropdown(
                    choices=[6, 9, 12, 15, 18, 24, 30],
                    value=9,
                    label="Video Duration (seconds)"
                )

                video_ratio = gr.Dropdown(
                    choices=list(RATIOS.keys()),
                    value="16:9",
                    label="Aspect Ratio"
                )

            video_create_button = gr.Button(
                "🚀 Create Video",
                variant="primary"
            )

            video_create_output = gr.Video(
                label="Generated Video"
            )

            video_create_download = gr.File(
                label="Download MP4"
            )

            video_create_button.click(
                fn=create_video,
                inputs=[
                    video_topic,
                    video_duration,
                    video_ratio
                ],
                outputs=[
                    video_create_output,
                    video_create_download
                ],
                api_name="create_video"
            )


# ============================================================
# QUEUE AND LAUNCH
# ============================================================

app.queue(
    max_size=40,
    default_concurrency_limit=3
)


if __name__ == "__main__":

    logger.info("ALPHA AI STARTING")

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
