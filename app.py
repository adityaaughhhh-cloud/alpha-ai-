import os
import io
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
# ALPHA AI 10.0
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
        "ALPHA_QUOTA_SALT must contain "
        "at least 32 characters."
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
            "Invalid quota response"
        )

    if "allowed" not in result:

        raise RuntimeError(
            "Missing allowed field"
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
# PROMPT BUILDER
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
                "Cloudflare generation failed."
            )

        encoded = data.get(
            "result",
            {}
        ).get("image")

        if not encoded:

            raise RuntimeError(
                "No image returned."
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
# GEMINI RESPONSE
# ============================================================

def ask_gemini(
    prompt
):

    response = gemini.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt,

        config=types.GenerateContentConfig(

            temperature=0.7,

            max_output_tokens=1200

        )

    )

    return response.text or (
        "Alpha could not generate a response."
    )


# ============================================================
# CHAT IMAGE DETECTION
# ============================================================

def wants_image(text):

    value = text.lower()

    keywords = [

        "generate an image",

        "create an image",

        "make an image",

        "draw an image",

        "generate image",

        "create image",

        "make image",

        "draw image",

        "create a picture",

        "generate a picture",

        "make a picture",

        "draw a picture"

    ]

    return any(
        word in value
        for word in keywords
    )


# ============================================================
# CHAT
# ============================================================

def chat_reply(
    message,
    history,
    request: gr.Request
):

    history = list(
        history or []
    )

    if not message or not message.strip():

        return history, None, None

    try:

        # IMAGE GENERATION

        if wants_image(message):

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

            return (
                history,
                image_path,
                image_path
            )

        # NORMAL CHAT

        contents = []

        for item in history[-12:]:

            if not isinstance(item, dict):
                continue

            role = item.get("role")

            content = item.get("content", "")

            if not isinstance(content, str):
                continue

            if role == "assistant":

                role = "model"

            if role not in ["user", "model"]:
                continue

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

        response = gemini.models.generate_content(

            model=GEMINI_MODEL,

            contents=contents,

            config=types.GenerateContentConfig(

                system_instruction=(
                    "You are Alpha AI, a helpful, "
                    "accurate and clear AI assistant. "
                    "Answer directly and explain when needed."
                ),

                temperature=0.7,

                max_output_tokens=1200

            )

        )

        answer = response.text or (
            "Alpha could not generate a response."
        )

        history.append({

            "role": "user",

            "content": message

        })

        history.append({

            "role": "assistant",

            "content": answer

        })

        return history, None, None

    except Exception:

        logger.exception(
            "CHAT ERROR"
        )

        history.append({

            "role": "assistant",

            "content": (
                "Alpha is temporarily unable to answer. "
                "Please try again."
            )

        })

        return history, None, None


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
            "Describe this image in detail. "
            "Explain its important objects and context."
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
            "new image. The requested transformation is: "

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
            "This is a text-guided generation, "
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

        mime_type = (
            mimetypes.guess_type(video_path)[0]
            or "video/mp4"
        )

        uploaded = gemini.files.upload(

            file=video_path

        )

        # WAIT FOR VIDEO PROCESSING

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

            + (description or "an important scene from the video")

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
                    "Ask Alpha anything or ask it to generate an image..."
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

            chat_send.click(

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

                outputs=[
                    chatbot,
                    chat_image,
                    chat_download
                ],

                api_name="chat_submit"

            ).then(

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
        "ALPHA AI 10.0 STARTING"
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
