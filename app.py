
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


# ==========================================
# ALPHA AI CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha")

PORT = int(os.getenv("PORT", "10000"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

GEMINI_BACKUP_MODEL = os.getenv(
    "GEMINI_BACKUP_MODEL",
    "gemini-3.5-flash-lite"
)

CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
)

client = None


# ==========================================
# GEMINI CLIENT
# ==========================================

def get_client():
    global client

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing in Render Environment."
        )

    if client is None:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=90000
            )
        )

    return client


# ==========================================
# FAST MODEL CONFIGURATION
# ==========================================

def get_generation_config(max_tokens=1024):
    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(
            thinking_level="low"
        )
    )


# ==========================================
# ERROR DETECTION
# ==========================================

def is_temporary_error(error):
    message = str(error).upper()

    codes = [
        "503",
        "UNAVAILABLE",
        "500",
        "502",
        "504",
        "RESOURCE EXHAUSTED",
        "SERVICE UNAVAILABLE",
        "INTERNAL SERVER ERROR",
        "BAD GATEWAY",
        "GATEWAY TIMEOUT",
        "429"
    ]

    return any(code in message for code in codes)


# ==========================================
# MODEL LIST
# ==========================================

def get_models():
    models = [GEMINI_MODEL]

    if GEMINI_BACKUP_MODEL != GEMINI_MODEL:
        models.append(GEMINI_BACKUP_MODEL)

    return models


# ==========================================
# NON-STREAMING FALLBACK
# ==========================================

def generate_with_fallback(contents, config=None):

    ai = get_client()
    last_error = None

    if config is None:
        config = get_generation_config()

    for model in get_models():

        try:
            logger.info("Trying Gemini model: %s", model)

            response = ai.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            logger.info("Gemini response received from %s", model)

            return response

        except Exception as error:
            last_error = error

            logger.exception(
                "Gemini model %s failed",
                model
            )

            if not is_temporary_error(error):
                raise

            logger.warning(
                "Switching to next available model."
            )

    raise RuntimeError(
        "All Gemini models failed. Last error: "
        + str(last_error)
    )


# ==========================================
# STREAMING CHAT WITH BACKUP
# ==========================================

def stream_chat_with_fallback(contents, config):

    ai = get_client()
    last_error = None

    for model in get_models():

        answer = ""

        try:
            logger.info("Starting chat with model: %s", model)

            stream = ai.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config
            )

            for chunk in stream:

                if chunk.text:
                    answer += chunk.text
                    yield answer

            if answer:
                logger.info(
                    "Chat completed using %s",
                    model
                )
                return

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        except Exception as error:

            last_error = error

            logger.exception(
                "Streaming failed using %s",
                model
            )

            if not is_temporary_error(error):
                raise

            # Clear incomplete primary response.
            if answer:
                yield ""

            logger.warning(
                "Trying backup model after temporary failure."
            )

    raise RuntimeError(
        "All Gemini models are unavailable. "
        + str(last_error)
    )


# ==========================================
# CHAT
# ==========================================

def chat(message, history):

    message = (message or "").strip()
    history = list(history or [])

    if not message:
        yield history
        return

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": ""
    })

    answer_index = len(history) - 1

    try:

        contents = []

        # Keep the latest 6 messages as context.
        previous_messages = history[:-2][-6:]

        for item in previous_messages:

            role = item.get("role")
            content = item.get("content")

            if role not in ("user", "assistant"):
                continue

            if not isinstance(content, str) or not content:
                continue

            contents.append(
                types.Content(
                    role=(
                        "user"
                        if role == "user"
                        else "model"
                    ),
                    parts=[
                        types.Part(text=content)
                    ]
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(text=message)
                ]
            )
        )

        config = get_generation_config(
            max_tokens=1024
        )

        for answer in stream_chat_with_fallback(
            contents,
            config
        ):

            history[answer_index]["content"] = answer
            yield history

    except Exception:
        logger.exception("Alpha chat error")

        history[answer_index]["content"] = (
            "Alpha is temporarily unable to answer. "
            "Please try again shortly."
        )

        yield history


# ==========================================
# IMAGE GENERATION
# ==========================================

def generate_image(prompt):

    prompt = (prompt or "").strip()

    if not prompt:
        raise gr.Error(
            "Please enter an image description."
        )

    if not CLOUDFLARE_API_TOKEN:
        raise gr.Error(
            "Cloudflare API token is missing."
        )

    if not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error(
            "Cloudflare account ID is missing."
        )

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + CLOUDFLARE_ACCOUNT_ID
        + "/ai/run/"
        + IMAGE_MODEL
    )

    try:

        response = requests.post(
            url,
            headers={
                "Authorization": (
                    "Bearer " + CLOUDFLARE_API_TOKEN
                ),
                "Content-Type": "application/json"
            },
            json={
                "prompt": prompt
            },
            timeout=90
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            ""
        )

        if "image/" in content_type:

            image_bytes = response.content

        else:

            data = response.json()

            encoded = data.get(
                "result", {}
            ).get("image")

            if not encoded:
                raise RuntimeError(str(data))

            image_bytes = base64.b64decode(
                encoded
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image.thumbnail((1280, 720))

        file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        image.save(
            file.name,
            format="PNG"
        )

        file.close()

        return file.name

    except Exception as error:

        logger.exception(
            "Image generation error"
        )

        raise gr.Error(
            f"Image generation failed: {error}"
        )


# ==========================================
# IMAGE ANALYSIS
# ==========================================

def analyze_image(image_path, question):

    if not image_path:
        raise gr.Error(
            "Upload an image first."
        )

    question = (
        (question or "").strip()
        or "Describe this image."
    )

    try:

        with open(image_path, "rb") as file:
            image_bytes = file.read()

        extension = image_path.lower()

        mime_type = "image/jpeg"

        if extension.endswith(".png"):
            mime_type = "image/png"

        elif extension.endswith(".webp"):
            mime_type = "image/webp"

        contents = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
            types.Part(text=question)
        ]

        response = generate_with_fallback(
            contents,
            get_generation_config(1200)
        )

        return (
            response.text
            or "No answer received."
        )

    except Exception as error:

        logger.exception(
            "Image analysis error"
        )

        raise gr.Error(
            f"Image analysis failed: {error}"
        )


# ==========================================
# VIDEO ANALYSIS
# ==========================================

def analyze_video(video_path, question):

    if not video_path:
        raise gr.Error(
            "Upload a video first."
        )

    question = (
        (question or "").strip()
        or "Summarize this video and explain its main events."
    )

    try:

        ai = get_client()

        logger.info("Uploading video to Gemini.")

        uploaded = ai.files.upload(
            file=video_path
        )

        deadline = time.time() + 180

        while (
            uploaded.state
            and str(uploaded.state).upper().endswith("PROCESSING")
        ):

            if time.time() > deadline:
                raise TimeoutError(
                    "Video processing timed out."
                )

            time.sleep(2)

            uploaded = ai.files.get(
                name=uploaded.name
            )

        contents = [
            uploaded,
            question
        ]

        response = generate_with_fallback(
            contents,
            get_generation_config(1500)
        )

        return (
            response.text
            or "No answer received."
        )

    except Exception as error:

        logger.exception(
            "Video analysis error"
        )

        raise gr.Error(
            f"Video analysis failed: {error}"
        )


# ==========================================
# ALPHA INTERFACE
# ==========================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
"""


with gr.Blocks(
    title="Alpha AI",
    theme=gr.themes.Soft(),
    css=CSS
) as demo:

    gr.Markdown("# Alpha AI")
    gr.Markdown("### Think Beyond Limits.")

    with gr.Tabs():

        # ----------------------------------
        # CHAT
        # ----------------------------------

        with gr.Tab("Chat"):

            chatbot = gr.Chatbot(
                label="Alpha Assistant",
                height=480
            )

            with gr.Row():

                message = gr.Textbox(
                    placeholder="Ask Alpha anything...",
                    label="Message",
                    scale=8
                )

                send = gr.Button(
                    "Send",
                    variant="primary"
                )

            clear = gr.Button("Clear chat")

            send.click(
                chat,
                inputs=[
                    message,
                    chatbot
                ],
                outputs=chatbot
            )

            message.submit(
                chat,
                inputs=[
                    message,
                    chatbot
                ],
                outputs=chatbot
            )

            clear.click(
                lambda: [],
                outputs=chatbot
            )

        # ----------------------------------
        # IMAGE STUDIO
        # ----------------------------------

        with gr.Tab("Image Studio"):

            prompt = gr.Textbox(
                label="Image prompt",
                placeholder="Describe the image you want..."
            )

            generate = gr.Button(
                "Generate Image",
                variant="primary"
            )

            generated = gr.Image(
                label="Generated Image",
                type="filepath"
            )

            generate.click(
                generate_image,
                inputs=prompt,
                outputs=generated
            )

        # ----------------------------------
        # IMAGE ANALYSIS
        # ----------------------------------

        with gr.Tab("Image Analysis"):

            image = gr.Image(
                label="Upload image",
                type="filepath"
            )

            image_question = gr.Textbox(
                label="Ask about the image",
                placeholder="What is in this image?"
            )

            image_button = gr.Button(
                "Analyze Image",
                variant="primary"
            )

            image_result = gr.Markdown()

            image_button.click(
                analyze_image,
                inputs=[
                    image,
                    image_question
                ],
                outputs=image_result
            )

        # ----------------------------------
        # VIDEO ANALYSIS
        # ----------------------------------

        with gr.Tab("Video Analysis"):

            video = gr.Video(
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

            video_result = gr.Markdown()

            video_button.click(
                analyze_video,
                inputs=[
                    video,
                    video_question
                ],
                outputs=video_result
            )


# ==========================================
# RENDER STARTUP
# ==========================================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=2
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT
)
    
