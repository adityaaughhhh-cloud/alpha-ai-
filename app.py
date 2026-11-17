import os
import io
import time
import base64
import logging
import tempfile
from pathlib import Path

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types

# ---------------- CONFIGURATION ----------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Alpha")

PORT = int(os.environ.get("PORT", "10000"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL", "gemini-2.5-flash"
)

IMAGE_MODEL = os.environ.get(
    "IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell"
)

MAX_HISTORY_MESSAGES = 10
MAX_OUTPUT_TOKENS = 1024

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ---------------- HELPERS ----------------

def check_gemini():
    if not client:
        raise gr.Error("GEMINI_API_KEY is missing in Render environment variables.")

def check_cloudflare():
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error("Cloudflare API token or account ID is missing.")

def clean_history(history):
    if not isinstance(history, list):
        return []

    cleaned = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str):
            cleaned.append({"role": role, "content": content})

    return cleaned

def friendly_error(error):
    logger.exception("Alpha request failed")
    return f"Alpha could not complete that request. Please try again. ({str(error)[:250]})"

# ---------------- CHAT ----------------

def chat(message, history):
    if not message or not message.strip():
        return clean_history(history), ""

    try:
        check_gemini()

        history = clean_history(history)

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
                temperature=0.7,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
        )

        answer = response.text or "I could not generate a response."

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        return history, ""

    except Exception as e:
        logger.exception("Chat error")
        history = clean_history(history)
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": friendly_error(e)
        })
        return history, ""

def clear_chat():
    return [], ""

# ---------------- IMAGE GENERATION ----------------

def generate_image(prompt, ratio):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an image description.")

    try:
        check_cloudflare()

        dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1024, 768),
            "3:4": (768, 1024)
        }

        width, height = dimensions.get(ratio, (1024, 1024))

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
        )

        headers = {
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt.strip(),
            "width": width,
            "height": height,
            "num_steps": 4
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if not response.ok:
            raise RuntimeError(
                f"Cloudflare error {response.status_code}: {response.text[:400]}"
            )

        content_type = response.headers.get("content-type", "")

        if "image/" in content_type:
            image_bytes = response.content
        else:
            data = response.json()

            if not data.get("success", True):
                raise RuntimeError(str(data.get("errors", data)))

            result = data.get("result", {})
            encoded = result.get("image")

            if not encoded:
                raise RuntimeError("Cloudflare did not return an image.")

            image_bytes = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        output_dir = Path(tempfile.gettempdir()) / "alpha_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"alpha_{int(time.time() * 1000)}.png"
        output_path = output_dir / filename

        image.save(output_path, format="PNG")

        return str(output_path), str(output_path)

    except Exception as e:
        raise gr.Error(f"Image generation failed: {str(e)[:300]}")

# ---------------- IMAGE ANALYSIS ----------------

def analyze_image(image_path, question):
    if not image_path:
        raise gr.Error("Please upload an image.")

    try:
        check_gemini()

        image = Image.open(image_path).convert("RGB")

        prompt = question.strip() if question and question.strip() else (
            "Describe this image clearly. Identify its main objects, "
            "details, and important information."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1200
            )
        )

        return response.text or "I could not analyze this image."

    except Exception as e:
        return friendly_error(e)

# ---------------- VIDEO ANALYSIS ----------------

def analyze_video(video_path, question):
    if not video_path:
        raise gr.Error("Please upload a video.")

    uploaded_file = None

    try:
        check_gemini()

        uploaded_file = client.files.upload(file=video_path)

        # Wait for Gemini video processing.
        for _ in range(60):
            state = getattr(uploaded_file.state, "name", "")
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini could not process this video.")

            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
        else:
            raise TimeoutError("Video processing took too long.")

        prompt = question.strip() if question and question.strip() else (
            "Analyze this video. Summarize what happens, describe the "
            "important scenes, and identify key details."
        )

        video_part = types.Part.from_uri(
            file_uri=uploaded_file.uri,
            mime_type=uploaded_file.mime_type
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, video_part],
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1500
            )
        )

        return response.text or "I could not analyze this video."

    except Exception as e:
        return friendly_error(e)

    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                logger.warning("Could not delete temporary Gemini video file.")

# ---------------- USER INTERFACE ----------------

with gr.Blocks(title="Alpha AI") as demo:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits.
        Chat, create images, and analyze images and videos.
        """
    )

    with gr.Tabs():

        # CHAT TAB
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=480,
                show_copy_button=True,
                placeholder="Welcome to Alpha AI. Ask me anything."
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask Alpha anything...",
                    scale=8,
                    show_label=False
                )
                send_button = gr.Button("Send", variant="primary", scale=1)

            clear_button = gr.Button("Clear Chat")

            send_button.click(
                chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )

            chat_input.submit(
                chat,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )

            clear_button.click(
                clear_chat,
                outputs=[chatbot, chat_input]
            )

        # IMAGE GENERATION TAB
        with gr.Tab("Image Studio"):
            gr.Markdown("### Create images with Alpha")

            image_prompt = gr.Textbox(
                label="Image description",
                placeholder="Describe the image you want to create...",
                lines=4
            )

            image_ratio = gr.Dropdown(
                choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                value="1:1",
                label="Image ratio"
            )

            generate_button = gr.Button(
                "Generate Image",
                variant="primary"
            )

            generated_image = gr.Image(
                label="Generated Image",
                type="filepath"
            )

            image_download = gr.File(label="Download Image")

            generate_button.click(
                generate_image,
                inputs=[image_prompt, image_ratio],
                outputs=[generated_image, image_download]
            )

        # IMAGE ANALYSIS TAB
        with gr.Tab("Image Analysis"):
            gr.Markdown("### Ask Alpha about an image")

            input_image = gr.Image(
                label="Upload Image",
                type="filepath"
            )

            image_question = gr.Textbox(
                label="Your question",
                placeholder="What is in this image?",
                lines=2
            )

            image_analyze_button = gr.Button(
                "Analyze Image",
                variant="primary"
            )

            image_result = gr.Textbox(
                label="Alpha's Analysis",
                lines=10
            )

            image_analyze_button.click(
                analyze_image,
                inputs=[input_image, image_question],
                outputs=image_result
            )

        # VIDEO ANALYSIS TAB
        with gr.Tab("Video Analysis"):
            gr.Markdown("### Upload a video and ask Alpha")

            input_video = gr.Video(label="Upload Video")

            video_question = gr.Textbox(
                label="Your question",
                placeholder="Summarize this video...",
                lines=2
            )

            video_analyze_button = gr.Button(
                "Analyze Video",
                variant="primary"
            )

            video_result = gr.Textbox(
                label="Alpha's Video Analysis",
                lines=12
            )

            video_analyze_button.click(
                analyze_video,
                inputs=[input_video, video_question],
                outputs=video_result
            )

    gr.Markdown(
        "---\n"
        "© Alpha AI | Think Beyond Limits."
    )

# ---------------- START SERVER ----------------

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True
    )
    
