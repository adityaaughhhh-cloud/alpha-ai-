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

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import requests

from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI 14.1
# ULTRA-FAST CHAT + IMAGE + VIDEO
# Gradio 6 compatibility fix
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

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BACKUP_MODEL = os.environ.get(
    "GEMINI_BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)

API_TIMEOUT = 120000
DAILY_IMAGE_LIMIT = 10
IMAGE_TIMEOUT = 120
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
    "GEMINI_API_KEY": GEMINI_API_KEY
}

missing = [name for name, value in required.items() if not value]

if missing:
    raise RuntimeError(
        "Missing environment variables: " + ", ".join(missing)
    )

if len(QUOTA_SALT) < 32:
    raise RuntimeError(
        "ALPHA_QUOTA_SALT must contain at least 32 characters."
    )


# ============================================================
# API CLIENTS
# ============================================================

http = requests.Session()

gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=API_TIMEOUT)
)

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    + CLOUDFLARE_ACCOUNT_ID
    + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
)


# ============================================================
# GEMINI RESPONSE FUNCTION
# ============================================================

def gemini_generate(contents, config, operation="Gemini"):
    models = []

    if GEMINI_MODEL:
        models.append(GEMINI_MODEL)

    if GEMINI_BACKUP_MODEL and GEMINI_BACKUP_MODEL not in models:
        models.append(GEMINI_BACKUP_MODEL)

    last_error = None

    for model_name in models:
        try:
            logger.info("%s: trying model %s", operation, model_name)

            response = gemini.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            answer = response.text

            if answer and answer.strip():
                logger.info(
                    "%s: successful response from %s",
                    operation,
                    model_name
                )
                return answer.strip()

            raise RuntimeError("Gemini returned an empty response.")

        except Exception as error:
            last_error = error
            logger.exception(
                "%s failed with model %s",
                operation,
                model_name
            )

    raise RuntimeError(
        operation + " failed with all configured models."
    ) from last_error


# ============================================================
# DATE
# ============================================================

def today():
    return datetime.now(IST).date().isoformat()


# ============================================================
# VISITOR IDENTIFICATION
# ============================================================

def get_visitor(request):
    ip = "unknown"

    try:
        if request and request.client:
            ip = request.client.host or "unknown"
    except Exception:
        logger.exception("VISITOR IDENTIFICATION ERROR")

    value = QUOTA_SALT + ":" + ip
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ============================================================
# SUPABASE
# ============================================================

def supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": "Bearer " + SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": "application/json"
    }


def consume_image_quota(visitor):
    endpoint = SUPABASE_URL + "/rest/v1/rpc/consume_alpha_media_quota"

    payload = {
        "p_visitor_hash": visitor,
        "p_usage_date": today(),
        "p_media_type": "image",
        "p_daily_limit": DAILY_IMAGE_LIMIT
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


def refund_image_quota(visitor, usage_date):
    endpoint = SUPABASE_URL + "/rest/v1/rpc/refund_alpha_media_quota"

    payload = {
        "p_visitor_hash": visitor,
        "p_usage_date": usage_date,
        "p_media_type": "image"
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
        logger.exception("IMAGE CREDIT REFUND FAILED")


# ============================================================
# IMAGE SETTINGS
# ============================================================

STYLES = {
    "Automatic": "",
    "Photorealistic": "Photorealistic image, realistic lighting and details.",
    "Cinematic": "Cinematic composition and dramatic lighting.",
    "Anime": "High quality anime illustration.",
    "3D Render": "Professional detailed 3D render.",
    "Digital Art": "Professional digital artwork.",
    "Watercolor": "Traditional watercolor painting.",
    "Oil Painting": "Detailed oil painting.",
    "Pixel Art": "High quality pixel art.",
    "Minimalist": "Clean minimalist artistic composition.",
    "Product Photography": "Professional studio product photography."
}

RATIOS = {
    "1:1": (720, 720),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (960, 720),
    "3:4": (720, 960),
    "3:2": (1080, 720),
    "2:3": (720, 1080),
    "21:9": (1680, 720)
}


# ============================================================
# IMAGE PROMPT
# ============================================================

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
        "Authorization": "Bearer " + CLOUDFLARE_API_TOKEN,
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt[:2048],
        "steps": 4
    }

    response = http.post(
        CLOUDFLARE_URL,
        headers=headers,
        json=payload,
        timeout=(10, IMAGE_TIMEOUT)
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "application/json" in content_type:
        data = response.json()

        if not data.get("success"):
            raise RuntimeError("Cloudflare generation failed.")

        encoded = data.get("result", {}).get("image")

        if not encoded:
            raise RuntimeError("No image returned.")

        image_bytes = base64.b64decode(encoded)
    else:
        image_bytes = response.content

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()

    return image_bytes


# ============================================================
# IMAGE EXPORT
# ============================================================

def export_image(image_bytes, ratio="16:9"):
    width, height = RATIOS.get(ratio, (1280, 720))

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

        result = ImageOps.pad(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            color=(15, 15, 20)
        )

        output = OUTPUT_DIR / (uuid.uuid4().hex + ".jpg")

        result.save(
            output,
            format="JPEG",
            quality=94
        )

        return str(output)


# ============================================================
# SHARED IMAGE GENERATION
# ============================================================

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
        raise RuntimeError("Request information unavailable.")

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

        final_prompt = build_prompt(prompt, style, extra)
        image_bytes = generate_image(final_prompt)

        return export_image(image_bytes, ratio)

    except Exception:
        if reserved:
            refund_image_quota(visitor, usage_date)
        raise


# ============================================================
# ULTRA-FAST LOCAL CALCULATOR
# NO GEMINI API CALL
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
    ast.UAdd: operator.pos
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

        if not isinstance(result, (int, float)):
            raise ValueError("Invalid result")

        if abs(result) > 10**15:
            raise ValueError("Result too large")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = FAST_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Invalid operation")

        return operation(calculate_expression(node.operand))

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
        "subtracted by": "-",
        "divided by": "/",
        "divide by": "/",
        "over": "/",
        "modulus": "%",
        "mod": "%"
    }

    for word, symbol in replacements.items():
        text = re.sub(
            r"\b" + re.escape(word) + r"\b",
            symbol,
            text
        )

    text = text.replace("×", "*")
    text = text.replace("÷", "/")
    text = text.replace("^", "**")
    text = text.replace(",", "")

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
# CHAT TOKEN ALLOCATION
# ============================================================

def choose_tokens(question):
    word_count = len(question.split())
    question_lower = question.lower()

    detailed_terms = [
        "explain in detail",
        "step by step",
        "compare",
        "difference between",
        "derive",
        "prove",
        "analyze",
        "comprehensive",
        "advanced",
        "detailed explanation"
    ]

    complex_question = any(
        term in question_lower for term in detailed_terms
    )

    if word_count <= 10 and not complex_question:
        return 512

    if word_count <= 30 and not complex_question:
        return 1024

    if word_count <= 70:
        return 2048

    return 4096


# ============================================================
# CHAT HISTORY
# ============================================================

def normalize_history(history):
    result = []

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            continue

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

        result.append({
            "role": role,
            "content": content
        })

    return result


# ============================================================
# ALPHA CHAT 14.1
# LOCAL FAST ANSWERS + GEMINI
# TEXT ONLY
# ============================================================

def chat_reply(message, history):
    history = normalize_history(history)

    if not message or not message.strip():
        return history

    message = message.strip()

    updated_history = history + [{
        "role": "user",
        "content": message
    }]

    fast_answer = fast_local_answer(message)

    if fast_answer is not None:
        logger.info("ALPHA FAST LOCAL CALCULATION SUCCESS")

        return updated_history + [{
            "role": "assistant",
            "content": fast_answer
        }]

    contents = []

    for item in history[-12:]:
        role = item["role"]

        if role == "assistant":
            role = "model"

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(text=item["content"])
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
            "You are Alpha AI, a helpful and accurate "
            "text-only AI assistant. "
            "Answer questions directly and clearly. "
            "For mathematics, show calculations. "
            "For science, explain concepts accurately. "
            "For simple questions, give short answers. "
            "For difficult questions, explain step by step. "
            "Never generate images or videos in Chat. "
            "Never call external image-generation services. "
            "Only return text."
        ),
        temperature=0.3,
        max_output_tokens=choose_tokens(message)
    )

    try:
        answer = gemini_generate(
            contents,
            config,
            operation="CHAT"
        )

        return updated_history + [{
            "role": "assistant",
            "content": answer
        }]

    except Exception:
        logger.exception("ALPHA CHAT FAILED")

        return updated_history + [{
            "role": "assistant",
            "content": (
                "Alpha is temporarily unable to answer. "
                "Please try again."
            )
        }]


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path, question):
    if not image_path:
        return "Please upload an image first."

    if not question or not question.strip():
        question = "Describe this image and its important details."

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
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
            operation="IMAGE ANALYSIS"
        )

    except Exception:
        logger.exception("IMAGE ANALYSIS ERROR")
        return "Image analysis failed. Please try again."


# ============================================================
# IMAGE-INSPIRED GENERATION
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
        path = create_image(prompt, request, ratio="16:9")
        return path, path

    except Exception:
        logger.exception("IMAGE VARIATION ERROR")
        return None, None


# ============================================================
# VIDEO PATH
# ============================================================

def get_video_path(video):
    if isinstance(video, str):
        return video

    if isinstance(video, dict):
        return video.get("video") or video.get("path")

    return None


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video, question):
    video_path = get_video_path(video)

    if not video_path:
        return "Please upload a video first."

    uploaded = None

    try:
        if not os.path.isfile(video_path):
            return "Video file not found."

        file_size = os.path.getsize(video_path)

        if file_size > VIDEO_MAX_SIZE:
            return "Video must be smaller than 50 MB."

        if file_size == 0:
            return "The video file is empty."

        uploaded = gemini.files.upload(file=video_path)

        for _ in range(90):
            uploaded = gemini.files.get(name=uploaded.name)

            state = getattr(uploaded, "state", None)
            state_name = getattr(state, "name", str(state)).upper()

            if state_name == "ACTIVE":
                break

            if state_name == "FAILED":
                raise RuntimeError("Video processing failed.")

            time.sleep(2)

        else:
            raise TimeoutError("Video processing timed out.")

        if not question or not question.strip():
            question = (
                "Summarize this video and explain "
                "its important visual scenes."
            )

        contents = [uploaded, question]

        config = types.GenerateContentConfig(
            max_output_tokens=1500,
            temperature=0.3
        )

        return gemini_generate(
            contents,
            config,
            operation="VIDEO ANALYSIS"
        )

    except Exception:
        logger.exception("VIDEO ANALYSIS ERROR")
        return "Video analysis failed. Please try again."

    finally:
        if uploaded is not None:
            try:
                gemini.files.delete(name=uploaded.name)
            except Exception:
                logger.exception("VIDEO FILE CLEANUP ERROR")


# ============================================================
# GENERATE IMAGE FROM VIDEO
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
        path = create_image(prompt, request, ratio="16:9")
        return path, path

    except Exception:
        logger.exception("VIDEO IMAGE GENERATION ERROR")
        return None, None


# ============================================================
# IMAGE STUDIO
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
        logger.exception("IMAGE STUDIO ERROR")
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

button {
    border-radius: 12px !important;
}

textarea {
    border-radius: 12px !important;
}

#result {
    border-radius: 16px;
    overflow: hidden;
}
"""


# ============================================================
# INTERFACE
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

        # ====================================================
        # CHAT
        # ====================================================

        with gr.Tab("💬 Chat"):

            # Gradio 6: do not pass the removed "type" argument.
            # Chatbot uses message dictionaries by default.
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

            studio_download = gr.File(label="Download Image")

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

            video_download = gr.File(label="Download Image")

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
    max_size=20,
    default_concurrency_limit=2
)


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    logger.info("ALPHA AI 14.1 STARTING")

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
