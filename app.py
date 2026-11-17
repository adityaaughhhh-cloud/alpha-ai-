import os
import io
import time
import logging
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types

# ============================================================
# ALPHA AI — Chat, image generation, image analysis, video analysis
# Required environment variables:
# GEMINI_API_KEY, CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID
# Optional: GEMINI_MODEL, IMAGE_MODEL, IMAGE_STEPS, PORT
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell").strip()
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "4"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
VIDEO_PROCESSING_TIMEOUT = int(os.getenv("VIDEO_PROCESSING_TIMEOUT", "180"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

_client = None


def get_gemini_client():
    global _client
    if not GEMINI_API_KEY:
        raise gr.Error("Gemini is not configured. Add GEMINI_API_KEY in Render Environment.")
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def require_cloudflare():
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error("Image generation is not configured. Add CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in Render Environment.")


def trim_history(history):
    if not isinstance(history, list):
        return []
    return history[-MAX_HISTORY_MESSAGES:]


def chat_reply(message, history):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return history, ""

    # Instant local arithmetic for simple expressions is intentionally omitted;
    # the model handles normal questions and explanations consistently.
    try:
        client = get_gemini_client()
        contents = []
        for pair in trim_history(history):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                old_user, old_assistant = pair
                if old_user:
                    contents.append(types.Content(role="user", parts=[types.Part(text=str(old_user))]))
                if old_assistant:
                    contents.append(types.Content(role="model", parts=[types.Part(text=str(old_assistant))]))
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are Alpha AI, a helpful, accurate assistant. "
                    "Answer clearly and naturally. Use simple language when appropriate. "
                    "Support emoji naturally. If unsure, say so instead of inventing facts."
                ),
                max_output_tokens=1024,
                temperature=0.7,
            ),
        )
        answer = (getattr(response, "text", None) or "I couldn't generate a response. Please try again.").strip()
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        answer = f"Alpha couldn't complete that request: {exc}"

    history.append((message, answer))
    return history, ""


def generate_image(prompt, aspect_ratio):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter an image description first.")
    require_cloudflare()

    dimensions = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1024, 768),
        "3:4": (768, 1024),
    }
    width, height = dimensions.get(aspect_ratio, (1024, 1024))
    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": IMAGE_STEPS,
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            if data.get("success") is False:
                raise RuntimeError(str(data.get("errors") or data))
            result = data.get("result", {})
            image_data = result.get("image") if isinstance(result, dict) else None
            if not image_data:
                raise RuntimeError(f"Unexpected Cloudflare response: {str(data)[:500]}")
            import base64
            raw = base64.b64decode(image_data)
        else:
            raw = response.content
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        output = Path(tempfile.gettempdir()) / f"alpha_image_{int(time.time() * 1000)}.png"
        image.save(output, format="PNG")
        return image, str(output)
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Image generation failed")
        raise gr.Error(f"Image generation failed: {exc}")


def analyze_image(image, question):
    if image is None:
        raise gr.Error("Upload an image first.")
    prompt = (question or "").strip() or "Describe and analyze this image in detail."
    try:
        client = get_gemini_client()
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(max_output_tokens=1200),
        )
        return getattr(response, "text", None) or "I couldn't analyze this image."
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise gr.Error(f"Image analysis failed: {exc}")


def analyze_video(video_path, question):
    if not video_path:
        raise gr.Error("Upload a video first.")
    client = get_gemini_client()
    uploaded_file = None
    try:
        uploaded_file = client.files.upload(file=video_path)
        deadline = time.time() + VIDEO_PROCESSING_TIMEOUT
        file_info = uploaded_file
        while time.time() < deadline:
            file_info = client.files.get(name=uploaded_file.name)
            state = getattr(file_info, "state", "")
            state_name = getattr(state, "name", str(state)).upper()
            if "ACTIVE" in state_name:
                break
            if "FAILED" in state_name:
                raise gr.Error("Gemini could not process this video.")
            time.sleep(2)
        else:
            raise gr.Error("Video processing timed out. Try a shorter video.")

        prompt = (question or "").strip() or "Summarize and analyze this video."
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[file_info, prompt],
            config=types.GenerateContentConfig(max_output_tokens=1400),
        )
        return getattr(response, "text", None) or "I couldn't analyze this video."
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Video analysis failed")
        raise gr.Error(f"Video analysis failed: {exc}")
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                logger.warning("Could not delete temporary Gemini video file.")


with gr.Blocks(title="Alpha AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Alpha AI\n### Think Beyond Limits.")
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=450, label="Alpha Assistant")
        with gr.Row():
            message = gr.Textbox(placeholder="Ask Alpha anything...", label="Message", scale=8)
            send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear chat")
        send.click(chat_reply, inputs=[message, chatbot], outputs=[chatbot, message])
        message.submit(chat_reply, inputs=[message, chatbot], outputs=[chatbot, message])
        clear.click(lambda: [], outputs=chatbot)

    with gr.Tab("Image Studio"):
        image_prompt = gr.Textbox(label="Describe your image", lines=3, placeholder="A cinematic view of a futuristic city at sunset...")
        aspect = gr.Dropdown(choices=["1:1", "16:9", "9:16", "4:3", "3:4"], value="1:1", label="Aspect ratio")
        make_image = gr.Button("Generate image", variant="primary")
        generated_image = gr.Image(label="Generated image", type="pil")
        image_file = gr.File(label="Download generated image")
        make_image.click(generate_image, inputs=[image_prompt, aspect], outputs=[generated_image, image_file])

    with gr.Tab("Image Analysis"):
        input_image = gr.Image(label="Upload an image", type="pil")
        image_question = gr.Textbox(label="What should Alpha analyze?", placeholder="Describe this image...")
        analyze_image_button = gr.Button("Analyze image", variant="primary")
        image_answer = gr.Markdown()
        analyze_image_button.click(analyze_image, inputs=[input_image, image_question], outputs=image_answer)

    with gr.Tab("Video Analysis"):
        input_video = gr.Video(label="Upload a video")
        video_question = gr.Textbox(label="What should Alpha analyze?", placeholder="Summarize this video...")
        analyze_video_button = gr.Button("Analyze video", variant="primary")
        video_answer = gr.Markdown()
        analyze_video_button.click(analyze_video, inputs=[input_video, video_question], outputs=video_answer)

    gr.Markdown("*Alpha AI may occasionally make mistakes. Verify important information.*")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
        
