import os
import io
import time
import base64
import logging
import tempfile

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types

# ==============================
# ALPHA CONFIGURATION
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaAI")

PORT = int(os.environ.get("PORT", "10000"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.environ.get(
    "IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell"
)

MAX_HISTORY = 8
MAX_TOKENS = 1024

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ==============================
# HELPER FUNCTIONS
# ==============================

def require_gemini():
    if client is None:
        raise gr.Error("Missing GEMINI_API_KEY in Render environment variables.")


def require_cloudflare():
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error(
            "Missing CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID."
        )


def safe_history(history):
    if not isinstance(history, list):
        return []

    result = []

    for item in history[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in ["user", "assistant"] and isinstance(content, str):
            result.append({
                "role": role,
                "content": content
            })

    return result


def error_text(error):
    logger.exception("Alpha error")
    return f"Request failed: {str(error)[:400]}"


def make_output_folder():
    folder = os.path.join(tempfile.gettempdir(), "alpha_ai_files")
    os.makedirs(folder, exist_ok=True)
    return folder


# ==============================
# CHAT
# ==============================

def chat_reply(message, history):
    if not message or not message.strip():
        return safe_history(history), ""

    try:
        require_gemini()

        history = safe_history(history)

        contents = []

        for item in history:
            contents.append(
                types.Content(
                    role=item["role"],
                    parts=[types.Part.from_text(text=item["content"])]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message.strip())]
            )
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.65,
                max_output_tokens=MAX_TOKENS
            )
        )

        answer = response.text or "Alpha couldn't generate an answer."

        history.append({"role": "user", "content": message.strip()})
        history.append({"role": "assistant", "content": answer})

        return history, ""

    except Exception as error:
        history = safe_history(history)
        history.append({"role": "user", "content": message.strip()})
        history.append({
            "role": "assistant",
            "content": error_text(error)
        })
        return history, ""


def clear_chat():
    return [], ""


# ==============================
# IMAGE GENERATION
# ==============================

def create_image(prompt, ratio):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter an image prompt first.")

    try:
        require_cloudflare()

        sizes = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1024, 768),
            "3:4": (768, 1024)
        }

        width, height = sizes.get(ratio, (1024, 1024))

        endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            + CLOUDFLARE_ACCOUNT_ID
            + "/ai/run/"
            + IMAGE_MODEL
        )

        headers = {
            "Authorization": "Bearer " + CLOUDFLARE_API_TOKEN,
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt.strip(),
            "width": width,
            "height": height,
            "num_steps": 4
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=120
        )

        if not response.ok:
            raise RuntimeError(
                f"Image API error {response.status_code}: "
                f"{response.text[:300]}"
            )

        content_type = response.headers.get("Content-Type", "")

        if content_type.startswith("image/"):
            image_bytes = response.content
        else:
            data = response.json()

            if not data.get("success", False):
                raise RuntimeError(str(data.get("errors", data)))

            image_data = data.get("result", {}).get("image")

            if not image_data:
                raise RuntimeError("No image data returned by API.")

            image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        filename = f"alpha_{int(time.time() * 1000)}.png"
        output_path = os.path.join(make_output_folder(), filename)

        image.save(output_path, format="PNG")

        return output_path, output_path

    except Exception as error:
        logger.exception("Image generation failed")
        raise gr.Error(str(error)[:400])


# ==============================
# IMAGE ANALYSIS
# ==============================

def analyze_image(image_path, question):
    if not image_path:
        return "Please upload an image first."

    try:
        require_gemini()

        image = Image.open(image_path).convert("RGB")

        prompt = question.strip() if question and question.strip() else (
            "Describe the image, its objects, scene, and important details."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1200
            )
        )

        return response.text or "No analysis was returned."

    except Exception as error:
        return error_text(error)


# ==============================
# VIDEO ANALYSIS
# ==============================

def analyze_video(video_path, question):
    if not video_path:
        return "Please upload a video first."

    uploaded = None

    try:
        require_gemini()

        uploaded = client.files.upload(file=video_path)

        for attempt in range(60):
            state = getattr(uploaded.state, "name", "")

            if state == "ACTIVE":
                break

            if state == "FAILED":
                raise RuntimeError("Video processing failed.")

            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        else:
            raise TimeoutError("Video processing timed out.")

        prompt = question.strip() if question and question.strip() else (
            "Summarize this video. Explain the main events, important "
            "visual details, and what happens in the clip."
        )

        video_part = types.Part.from_uri(
            file_uri=uploaded.uri,
            mime_type=uploaded.mime_type
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, video_part],
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1500
            )
        )

        return response.text or "No video analysis returned."

    except Exception as error:
        return error_text(error)

    finally:
        if uploaded:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                logger.warning("Temporary video cleanup failed.")


# ==============================
# ALPHA INTERFACE
# ==============================

with gr.Blocks(title="Alpha AI") as demo:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits.

        Chat • Image Generation • Image Analysis • Video Analysis
        """
    )

    with gr.Tabs():

        # -------- CHAT --------

        with gr.Tab("Chat"):

            chatbot = gr.Chatbot(
                height=450,
                placeholder="Welcome to Alpha. Start a conversation."
            )

            with gr.Row():

                message = gr.Textbox(
                    placeholder="Ask Alpha anything...",
                    show_label=False,
                    scale=8
                )

                send = gr.Button("Send", variant="primary", scale=1)

            clear = gr.Button("Clear conversation")

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
                clear_chat,
                inputs=None,
                outputs=[chatbot, message]
            )

        # -------- IMAGE STUDIO --------

        with gr.Tab("Image Studio"):

            gr.Markdown("## Create an image")

            prompt = gr.Textbox(
                label="Describe your image",
                placeholder="A futuristic city on Mars...",
                lines=4
            )

            ratio = gr.Dropdown(
                choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                value="1:1",
                label="Aspect ratio"
            )

            generate = gr.Button("Generate", variant="primary")

            image_preview = gr.Image(
                label="Generated image",
                type="filepath"
            )

            image_file = gr.File(label="Download generated image")

            generate.click(
                create_image,
                inputs=[prompt, ratio],
                outputs=[image_preview, image_file]
            )

        # -------- IMAGE ANALYSIS --------

        with gr.Tab("Image Analysis"):

            gr.Markdown("## Analyze an image")

            uploaded_image = gr.Image(
                label="Upload image",
                type="filepath"
            )

            image_question = gr.Textbox(
                label="Question",
                placeholder="Explain this image...",
                lines=2
            )

            image_button = gr.Button("Analyze", variant="primary")

            image_answer = gr.Textbox(
                label="Alpha's answer",
                lines=10
            )

            image_button.click(
                analyze_image,
                inputs=[uploaded_image, image_question],
                outputs=image_answer
            )

        # -------- VIDEO ANALYSIS --------

        with gr.Tab("Video Analysis"):

            gr.Markdown("## Analyze a video")

            uploaded_video = gr.Video(label="Upload video")

            video_question = gr.Textbox(
                label="Question",
                placeholder="Summarize this video...",
                lines=2
            )

            video_button = gr.Button("Analyze video", variant="primary")

            video_answer = gr.Textbox(
                label="Alpha's video analysis",
                lines=12
            )

            video_button.click(
                analyze_video,
                inputs=[uploaded_video, video_question],
                outputs=video_answer
            )

    gr.Markdown("---\n**Alpha AI · Think Beyond Limits**")


# ==============================
# START
# ==============================

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=PORT
    )
    
