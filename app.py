import os
import io
import re
import uuid
import time
import base64
import hashlib
import logging
import mimetypes

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import requests

from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI 11.0
# CHAT + IMAGE STUDIO + IMAGE ANALYSIS + VIDEO ANALYSIS
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ALPHA")


# ============================================================
# CONFIGURATION
# ============================================================

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

QUOTA_SALT = os.environ.get(
    "ALPHA_QUOTA_SALT"
)

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

DAILY_IMAGE_LIMIT = 10

IMAGE_TIMEOUT = 120

VIDEO_MAX_SIZE = 50 * 1024 * 1024

PORT = int(
    os.environ.get("PORT", "10000")
)

IST = ZoneInfo("Asia/Kolkata")

OUTPUT_DIR = Path("/tmp/alpha-images")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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

missing = [
    name
    for name, value in required.items()
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

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    + CLOUDFLARE_ACCOUNT_ID
    + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
)


# ============================================================
# DATE
# ============================================================

def today():

    return datetime.now(
        IST
    ).date().isoformat()


# ============================================================
# VISITOR IDENTIFICATION
# ============================================================

def get_visitor(request):

    ip = "unknown"

    try:

        if request and request.client:

            ip = request.client.host or "unknown"

    except Exception:

        logger.exception(
            "VISITOR IDENTIFICATION ERROR"
        )

    value = QUOTA_SALT + ":" + ip

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


# ============================================================
# SUPABASE HEADERS
# ============================================================

def supabase_headers():

    return {

        "apikey": SUPABASE_SERVICE_ROLE_KEY,

        "Authorization": (
            "Bearer " + SUPABASE_SERVICE_ROLE_KEY
        ),

        "Content-Type": "application/json"

    }


# ============================================================
# CONSUME IMAGE QUOTA
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

        "p_daily_limit": DAILY_IMAGE_LIMIT

    }

    response = http.post(

        endpoint,

        headers=supabase_headers(),

        json=payload,

        timeout=20

    )

    if not response.ok:

        logger.error(
            "SUPABASE QUOTA ERROR: %s %s",
            response.status_code,
            response.text[:1000]
        )

        response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):

        raise RuntimeError(
            "Invalid quota response."
        )

    if "allowed" not in result:

        raise RuntimeError(
            "Missing allowed field in quota response."
        )

    return result


# ============================================================
# REFUND IMAGE QUOTA
# ============================================================

def refund_image_quota(
    visitor,
    usage_date
):

    endpoint = (
        SUPABASE_URL
        + "/rest/v1/rpc/refund_alpha_media_quota"
    )

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

        logger.exception(
            "IMAGE CREDIT REFUND FAILED"
        )


# ============================================================
# IMAGE STYLES
# ============================================================

STYLES = {

    "Automatic": "",

    "Photorealistic":
        "Photorealistic image, realistic lighting and natural details.",

    "Cinematic":
        "Cinematic composition, dramatic lighting and film quality.",

    "Anime":
        "High quality anime illustration.",

    "3D Render":
        "Professional detailed 3D render.",

    "Digital Art":
        "Professional digital artwork.",

    "Watercolor":
        "Traditional watercolor painting.",

    "Oil Painting":
        "Traditional detailed oil painting.",

    "Pixel Art":
        "High quality pixel art.",

    "Minimalist":
        "Clean minimalist artistic composition.",

    "Product Photography":
        "Professional studio product photography."

}


# ============================================================
# IMAGE RATIOS
# ============================================================

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
# IMAGE PROMPT BUILDER
# ============================================================

def build_prompt(
    prompt,
    style="Automatic",
    extra=""
):

    style_text = STYLES.get(
        style,
        ""
    )

    final_prompt = (

        prompt.strip()

        + "\n"

        + style_text

        + "\n"

        + (extra or "")

    )

    return final_prompt[:2048]


# ============================================================
# CLOUDFLARE IMAGE GENERATION
# ============================================================

def generate_image(prompt):

    headers = {

        "Authorization": (
            "Bearer " + CLOUDFLARE_API_TOKEN
        ),

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

    if not response.ok:

        logger.error(
            "CLOUDFLARE ERROR: %s %s",
            response.status_code,
            response.text[:1000]
        )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if "application/json" in content_type:

        data = response.json()

        if not data.get("success"):

            raise RuntimeError(
                "Cloudflare image generation failed."
            )

        encoded = data.get(
            "result",
            {}
        ).get("image")

        if not encoded:

            raise RuntimeError(
                "No image returned by Cloudflare."
            )

        image_bytes = base64.b64decode(
            encoded
        )

    else:

        image_bytes = response.content

    with Image.open(
        io.BytesIO(image_bytes)
    ) as image:

        image.verify()

    return image_bytes


# ============================================================
# IMAGE EXPORT
# ============================================================

def export_image(
    image_bytes,
    ratio="16:9"
):

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

            color=(15, 15, 20)

        )

        output = OUTPUT_DIR / (
            uuid.uuid4().hex + ".jpg"
        )

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

        raise ValueError(
            "Please enter an image description."
        )

    if request is None:

        raise RuntimeError(
            "Request information unavailable."
        )

    visitor = get_visitor(request)

    usage_date = today()

    reserved = False

    try:

        quota = consume_image_quota(
            visitor
        )

        if not quota.get("allowed"):

            raise RuntimeError(
                "Daily image limit reached. "
                "You can create 10 images per day."
            )

        reserved = True

        final_prompt = build_prompt(

            prompt,

            style,

            extra

        )

        image_bytes = generate_image(
            final_prompt
        )

        output_path = export_image(

            image_bytes,

            ratio

        )

        return output_path

    except Exception:

        if reserved:

            refund_image_quota(
                visitor,
                usage_date
            )

        raise


# ============================================================
# DYNAMIC TOKEN ALLOCATION
# ============================================================

def choose_tokens(question):

    text = question.lower()

    words = len(
        text.split()
    )

    complex_topics = [

        "quantum",

        "relativity",

        "thermodynamics",

        "mechanics",

        "calculus",

        "derivation",

        "proof",

        "compare",

        "difference between",

        "step by step",

        "detailed",

        "comprehensive",

        "analyze",

        "explain in detail",

        "research",

        "advanced",

        "mathematical",

        "equation",

        "theory"

    ]

    complexity = sum(

        1 for topic in complex_topics

        if topic in text

    )

    if words <= 12 and complexity == 0:

        return 512

    if words <= 35 and complexity <= 1:

        return 1024

    if words <= 80 and complexity <= 2:

        return 2048

    return 4096


# ============================================================
# GEMINI GENERATION CONFIG
# ============================================================

def get_chat_config(question):

    token_limit = choose_tokens(
        question
    )

    return types.GenerateContentConfig(

        system_instruction=(

            "You are Alpha AI, a highly capable and "
            "accurate AI assistant. "

            "Answer the user's actual question. "

            "Use clear, simple language when appropriate. "

            "For difficult topics, explain step by step. "

            "Do not generate images unless the user "
            "explicitly requests image creation. "

            "Do not claim to have performed actions "
            "that you have not performed."

        ),

        temperature=0.5,

        max_output_tokens=token_limit

    )


# ============================================================
# STRICT IMAGE INTENT DETECTION
# ============================================================

def wants_image(text):

    if not text:

        return False

    value = text.lower().strip()

    patterns = [

        r"\b(generate|create|make|draw|design|produce)\b.{0,70}\b(image|picture|photo|illustration|artwork)\b",

        r"\b(show me|give me)\b.{0,40}\b(image|picture|photo)\b",

        r"\b(image|picture|photo)\s+of\s+"

    ]

    return any(

        re.search(pattern, value)

        for pattern in patterns

    )


# ============================================================
# CHAT HISTORY NORMALIZATION
# ============================================================

def normalize_history(history):

    result = []

    for item in history or []:

        if isinstance(item, dict):

            role = item.get("role")

            content = item.get("content", "")

        else:

            role = getattr(
                item,
                "role",
                None
            )

            content = getattr(
                item,
                "content",
                ""
            )

        if role not in ["user", "assistant", "model"]:

            continue

        if not isinstance(content, str):

            continue

        if not content.strip():

            continue

        if role == "model":

            role = "assistant"

        result.append({

            "role": role,

            "content": content

        })

    return result


# ============================================================
# CHAT CONTENT BUILDER
# ============================================================

def build_chat_contents(
    history,
    message
):

    contents = []

    for item in history[-12:]:

        role = item["role"]

        content = item["content"]

        if role == "assistant":

            role = "model"

        contents.append(

            types.Content(

                role=role,

                parts=[

                    types.Part.from_text(
                        text=content
                    )

                ]

            )

        )

    contents.append(

        types.Content(

            role="user",

            parts=[

                types.Part.from_text(
                    text=message
                )

            ]

        )

    )

    return contents


# ============================================================
# CHAT
# ============================================================

def chat_reply(
    message,
    history,
    request: gr.Request
):

    history = normalize_history(
        history
    )

    if not message or not message.strip():

        yield history, None, None

        return

    message = message.strip()

    # --------------------------------------------------------
    # IMAGE GENERATION ONLY WHEN EXPLICITLY REQUESTED
    # --------------------------------------------------------

    if wants_image(message):

        try:

            image_path = create_image(

                message,

                request,

                ratio="16:9"

            )

            history.append({

                "role": "user",

                "content": message

            })

            history.append({

                "role": "assistant",

                "content": "Your image has been generated."

            })

            yield history, image_path, image_path

            return

        except Exception as error:

            logger.exception(
                "CHAT IMAGE GENERATION ERROR"
            )

            history.append({

                "role": "user",

                "content": message

            })

            history.append({

                "role": "assistant",

                "content": (
                    "Image generation failed: "
                    + str(error)[:250]
                )

            })

            yield history, None, None

            return

    # --------------------------------------------------------
    # NORMAL TEXT CHAT
    # --------------------------------------------------------

    history.append({

        "role": "user",

        "content": message

    })

    contents = build_chat_contents(

        history[:-1],

        message

    )

    config = get_chat_config(
        message
    )

    answer = ""

    try:

        # STREAMING RESPONSE

        stream = gemini.models.generate_content_stream(

            model=GEMINI_MODEL,

            contents=contents,

            config=config

        )

        for chunk in stream:

            chunk_text = getattr(
                chunk,
                "text",
                None
            )

            if chunk_text:

                answer += chunk_text

                temporary_history = history + [{

                    "role": "assistant",

                    "content": answer

                }]

                yield temporary_history, None, None

        if not answer.strip():

            answer = (
                "Alpha could not generate a response. "
                "Please try again."
            )

        history.append({

            "role": "assistant",

            "content": answer

        })

        yield history, None, None

    except Exception:

        logger.exception(
            "CHAT ERROR"
        )

        error_history = history + [{

            "role": "assistant",

            "content": (
                "Alpha is temporarily unable to answer. "
                "Please try again."
            )

        }]

        yield error_history, None, None


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path,
    question
):

    if not image_path:

        return (
            "Please upload an image first."
        )

    if not question or not question.strip():

        question = (
            "Describe this image. Explain its "
            "important objects and visual details."
        )

    try:

        with Image.open(
            image_path
        ) as image:

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

        response = gemini.models.generate_content(

            model=GEMINI_MODEL,

            contents=[

                question,

                types.Part.from_bytes(

                    data=image_bytes,

                    mime_type="image/jpeg"

                )

            ],

            config=types.GenerateContentConfig(

                max_output_tokens=1200

            )

        )

        return response.text or (
            "No analysis returned."
        )

    except Exception:

        logger.exception(
            "IMAGE ANALYSIS ERROR"
        )

        return (
            "Image analysis failed. Please try again."
        )


# ============================================================
# IMAGE-INSPIRED GENERATION
# ============================================================

def generate_from_image_analysis(
    image_path,
    description,
    request: gr.Request
):

    if not image_path:

        return None, None

    if not description or not description.strip():

        return None, None

    try:

        visual_prompt = (

            "Study the uploaded image and create "
            "a detailed visual description for a "
            "new image. Requested transformation: "

            + description

        )

        analysis = analyze_image(

            image_path,

            visual_prompt

        )

        if (
            not analysis
            or "analysis failed" in analysis.lower()
        ):

            return None, None

        final_prompt = (

            "Create a new high-quality image using "
            "the following visual description. "
            "This is text-guided generation, "
            "not an exact edit of the uploaded image.\n\n"

            + description

            + "\n\n"

            + analysis

        )

        path = create_image(

            final_prompt,

            request,

            ratio="16:9"

        )

        return path, path

    except Exception:

        logger.exception(
            "IMAGE GENERATION FROM ANALYSIS ERROR"
        )

        return None, None


# ============================================================
# VIDEO PATH
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

    video_path = get_video_path(
        video
    )

    if not video_path:

        return (
            "Please upload a video first."
        )

    uploaded = None

    try:

        if not os.path.isfile(video_path):

            return (
                "The uploaded video file could not be found."
            )

        file_size = os.path.getsize(
            video_path
        )

        if file_size > VIDEO_MAX_SIZE:

            return (
                "Video is too large. Please upload "
                "a video smaller than 50 MB."
            )

        if file_size == 0:

            return (
                "The uploaded video is empty."
            )

        uploaded = gemini.files.upload(

            file=video_path

        )

        # WAIT FOR PROCESSING

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
                    "Gemini video processing failed."
                )

            time.sleep(2)

        else:

            raise TimeoutError(
                "Video processing timed out."
            )

        if not question or not question.strip():

            question = (

                "Analyze this video. Summarize what happens, "
                "identify important scenes, and explain "
                "the main visual content."

            )

        response = gemini.models.generate_content(

            model=GEMINI_MODEL,

            contents=[

                uploaded,

                question

            ],

            config=types.GenerateContentConfig(

                max_output_tokens=1500

            )

        )

        return response.text or (
            "No video analysis returned."
        )

    except Exception:

        logger.exception(
            "VIDEO ANALYSIS ERROR"
        )

        return (
            "Video analysis failed. Please check "
            "the video format and try again."
        )

    finally:

        if uploaded is not None:

            try:

                gemini.files.delete(

                    name=uploaded.name

                )

            except Exception:

                logger.exception(
                    "GEMINI FILE CLEANUP ERROR"
                )


# ============================================================
# GENERATE IMAGE FROM VIDEO ANALYSIS
# ============================================================

def generate_from_video(
    video,
    description,
    request: gr.Request
):

    if not get_video_path(video):

        return None, None

    try:

        question = (

            "Analyze this video and describe its "
            "important visual scenes for image creation. "
            "The user wants an image showing: "

            + (
                description
                or "an important scene from the video"
            )

        )

        analysis = analyze_video(

            video,

            question

        )

        failure_messages = [

            "analysis failed",

            "too large",

            "timed out",

            "could not be found",

            "empty"

        ]

        if any(

            word in analysis.lower()

            for word in failure_messages

        ):

            return None, None

        final_prompt = (

            "Create a new high-quality still image. "
            "Use the following video analysis as "
            "visual inspiration. This is not an "
            "extracted original video frame.\n\n"

            + (description or "")

            + "\n\n"

            + analysis

        )

        path = create_image(

            final_prompt,

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

            chatbot = gr.Chatbot(

                label="Alpha AI",

                height=480

            )

            chat_input = gr.Textbox(

                placeholder=(
                    "Ask Alpha anything. Request an image "
                    "only when you want one."
                ),

                label="Message",

                lines=2

            )

            chat_send = gr.Button(

                "Send",

                variant="primary"

            )

            chat_image = gr.Image(

                label="Generated Image",

                type="filepath",

                interactive=False

            )

            chat_download = gr.File(

                label="Download Generated Image"

            )

            chat_event = chat_send.click(

                fn=chat_reply,

                inputs=[

                    chat_input,

                    chatbot

                ],

                outputs=[

                    chatbot,

                    chat_image,

                    chat_download

                ],

                api_name="chat"

            )

            chat_event.then(

                fn=lambda: "",

                outputs=chat_input

            )

            submit_event = chat_input.submit(

                fn=chat_reply,

                inputs=[

                    chat_input,

                    chatbot

                ],

                outputs=[

                    chatbot,

                    chat_image,

                    chat_download

                ],

                api_name="chat_submit"

            )

            submit_event.then(

                fn=lambda: "",

                outputs=chat_input

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

    max_size=20,

    default_concurrency_limit=2

)


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    logger.info(
        "ALPHA AI 11.0 STARTING"
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        show_error=False,

        css=CSS,

        theme=gr.themes.Soft(

            primary_hue="blue",

            secondary_hue="slate"

        )

)
