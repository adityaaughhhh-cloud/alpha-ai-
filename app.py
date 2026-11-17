
import os
import io
import time
import uuid
import base64
import logging
import tempfile
import mimetypes
from pathlib import Path

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types

# ============================================================
# ALPHA AI
# Chat | Image Studio | Image Analysis | Video Analysis
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CF_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
)

IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "4"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
VIDEO_TIMEOUT = int(os.getenv("VIDEO_PROCESSING_TIMEOUT", "240"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "700"))

_client = None
http = requests.Session()

TEMP_DIR = Path(tempfile.gettempdir()) / "alpha"
TEMP_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# CLIENTS
# ============================================================

def get_client():
    global _client

    if not GEMINI_API_KEY:
        raise gr.Error("Add GEMINI_API_KEY in Render Environment.")

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def check_cloudflare():
    if not CF_TOKEN or not CF_ACCOUNT_ID:
        raise gr.Error(
            "Add CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID."
        )


def error_text(exc):
    return str(exc).strip()[:400]


# ============================================================
# FAST CHAT
# ============================================================

def chat_reply(message, history):

    message = (message or "").strip()
    history = history or []

    if not message:
        return history, ""

    try:
        client = get_client()

        contents = []

        for item in history[-MAX_HISTORY:]:

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
                    parts=[types.Part(text=content[:3500])]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are Alpha AI, a capable and accurate AI assistant. "
                    "Answer directly. Keep simple answers short. "
                    "Explain difficult subjects clearly. "
                    "Use simple English when useful. "
                    "Never invent facts. Admit uncertainty."
                ),
                temperature=0.4,
                max_output_tokens=MAX_TOKENS
            )
        )

        answer = (getattr(response, "text", None) or "").strip()

        if not answer:
            answer = "I couldn't generate an answer. Please try again."

    except Exception as exc:
        logger.exception("Chat failed")
        answer = "Alpha error: " + error_text(exc)

    history = list(history)

    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ])

    return history[-20:], ""


# ============================================================
# IMAGE STUDIO
# ============================================================

def generate_image(prompt, ratio, detail):

    prompt = (prompt or "").strip()

    if not prompt:
        raise gr.Error("Describe your image first.")

    check_cloudflare()

    sizes = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1024, 768),
        "3:4": (768, 1024)
    }

    width, height = sizes.get(ratio, (1024, 1024))

    detail_instruction = {
        "Standard": "Clean composition and balanced lighting.",
        "High detail": (
            "Highly detailed textures, professional composition, "
            "realistic lighting, sharp focus and refined color grading."
        )
    }.get(detail, "Professional visual quality.")

    enhanced_prompt = (
        prompt
        + ". Beautiful composition, clear main subject, "
        "consistent perspective, realistic proportions, "
        "detailed textures, balanced colors, professional lighting. "
        + detail_instruction
        + " Avoid visual distortions, duplicate objects, "
        "watermarks, signatures and unwanted text."
    )

    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CF_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )

    payload = {
        "prompt": enhanced_prompt,
        "width": width,
        "height": height,
        "num_steps": IMAGE_STEPS
    }

    headers = {
        "Authorization": f"Bearer {CF_TOKEN}"
    }

    try:

        response = http.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=(15, REQUEST_TIMEOUT)
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type", ""
        ).lower()

        if "image/" in content_type:

            raw = response.content

        else:

            data = response.json()

            if data.get("success") is False:
                raise RuntimeError(str(data.get("errors")))

            result = data.get("result") or {}
            encoded = result.get("image") if isinstance(result, dict) else None

            if not encoded:
                raise RuntimeError(
                    "Image missing from Cloudflare response."
                )

            raw = base64.b64decode(encoded)

        output = TEMP_DIR / f"alpha_{uuid.uuid4().hex}.png"

        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image.convert("RGB").save(
                output,
                format="PNG",
                optimize=True
            )

        return str(output), str(output)

    except requests.Timeout:
        raise gr.Error("Image generation timed out. Try again.")

    except Exception as exc:
        logger.exception("Image generation failed")
        raise gr.Error("Image error: " + error_text(exc))


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image_path, question):

    if not image_path:
        raise gr.Error("Upload an image first.")

    prompt = (question or "").strip()

    if not prompt:
        prompt = (
            "Analyze this image carefully. Describe its main subject, "
            "objects, scene, visible text and important details. "
            "Do not invent information."
        )

    try:

        client = get_client()

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1200
                )
            )

        return (
            getattr(response, "text", None)
            or "I couldn't analyze this image."
        )

    except Exception as exc:
        logger.exception("Image analysis failed")
        raise gr.Error("Image analysis error: " + error_text(exc))


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path, question):

    if not video_path:
        raise gr.Error("Upload a video first.")

    client = get_client()
    uploaded = None

    try:

        mime = (
            mimetypes.guess_type(str(video_path))[0]
            or "video/mp4"
        )

        uploaded = client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(mime_type=mime)
        )

        deadline = time.time() + VIDEO_TIMEOUT

        file_info = uploaded

        while time.time() < deadline:

            file_info = client.files.get(
                name=uploaded.name
            )

            state = getattr(file_info, "state", "")
            state_name = getattr(
                state, "name", str(state)
            ).upper()

            if "ACTIVE" in state_name:
                break

            if "FAILED" in state_name:
                raise RuntimeError("Video processing failed.")

            time.sleep(2)

        else:
            raise TimeoutError(
                "Video processing timed out. Try a shorter video."
            )

        prompt = (question or "").strip()

        if not prompt:
            prompt = (
                "Summarize this video chronologically. "
                "Explain the important events, visible objects "
                "and scene changes. Do not invent details."
            )

        video_part = types.Part.from_uri(
            file_uri=file_info.uri,
            mime_type=getattr(file_info, "mime_type", mime)
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1400
            )
        )

        return (
            getattr(response, "text", None)
            or "I couldn't analyze this video."
        )

    except Exception as exc:
        logger.exception("Video analysis failed")
        raise gr.Error("Video analysis error: " + error_text(exc))

    finally:

        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                logger.warning("Temporary video cleanup failed.")


# ============================================================
# ALPHA INTERFACE
# ============================================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#alpha-header {
    text-align: center;
    padding: 12px;
}

#alpha-header h1 {
    font-size: 36px;
    font-weight: 800;
}

#alpha-header p {
    opacity: 0.7;
}
"""

with gr.Blocks(
    title="Alpha AI",
    theme=gr.themes.Soft(),
    css=CSS
) as demo:

    gr.HTML("""
    <div id="alpha-header">
        <h1>ALPHA AI</h1>
        <p>Think Beyond Limits.</p>
    </div>
    """)

    # CHAT
    with gr.Tab("Chat"):

        chatbot = gr.Chatbot(
            label="Alpha Assistant",
            type="messages",
            height=470,
            show_copy_button=True
        )

        message = gr.Textbox(
            label="Message Alpha",
            placeholder="Ask Alpha anything...",
            lines=2
        )

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear chat")

        send.click(
            chat_reply,
            inputs=[message, chatbot],
            outputs=[chatbot, message]
        )

        message.submit(
            chat_reply,
            inputs=[message, chatbot],
            outputs=[chatbot, message]
        )

        clear.click(
            lambda: ([], ""),
            outputs=[chatbot, message]
        )

    # IMAGE STUDIO
    with gr.Tab("Image Studio"):

        gr.Markdown("### Create images with Alpha")

        image_prompt = gr.Textbox(
            label="Image description",
            placeholder="Describe the image you want...",
            lines=4
        )

        with gr.Row():

            ratio = gr.Dropdown(
                choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                value="1:1",
                label="Aspect ratio"
            )

            detail = gr.Dropdown(
                choices=["Standard", "High detail"],
                value="High detail",
                label="Image quality"
            )

        make_image = gr.Button(
            "Generate Image",
            variant="primary"
        )

        generated_image = gr.Image(
            label="Generated image",
            type="filepath",
            interactive=False
        )

        image_file = gr.File(
            label="Download image"
        )

        make_image.click(
            generate_image,
            inputs=[image_prompt, ratio, detail],
            outputs=[generated_image, image_file]
        )

    # IMAGE ANALYSIS
    with gr.Tab("Image Analysis"):

        input_image = gr.Image(
            label="Upload image",
            type="filepath"
        )

        image_question = gr.Textbox(
            label="Ask about the image",
            placeholder="What is happening here?"
        )

        image_button = gr.Button(
            "Analyze Image",
            variant="primary"
        )

        image_answer = gr.Markdown()

        image_button.click(
            analyze_image,
            inputs=[input_image, image_question],
            outputs=image_answer
        )

    # VIDEO ANALYSIS
    with gr.Tab("Video Analysis"):

        input_video = gr.Video(
            label="Upload video",
            sources=["upload"]
        )

        video_question = gr.Textbox(
            label="Ask about the video",
            placeholder="Summarize this video..."
        )

        video_button = gr.Button(
            "Analyze Video",
            variant="primary"
        )

        video_answer = gr.Markdown()

        video_button.click(
            analyze_video,
            inputs=[input_video, video_question],
            outputs=video_answer
        )

    gr.Markdown(
        "*Alpha AI may occasionally make mistakes. Verify important information.*"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", "10000"))

    demo.queue(
        default_concurrency_limit=4,
        max_size=20
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
    
