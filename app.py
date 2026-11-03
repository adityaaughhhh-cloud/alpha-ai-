import os
import io
import time
import base64
import tempfile
import logging
import random

import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# ==================================================
# ALPHA AI 3.7 - SPEED & STABILITY EDITION
# ==================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALPHA")


# ==================================================
# API CONFIGURATION
# ==================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")

client = genai.Client(api_key=API_KEY)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

CHAT_MODEL = "gemini-3.5-flash-lite"
BACKUP_MODEL = "gemini-3.1-flash-lite"

# Updated image model
IMAGE_MODEL = "gemini-3.1-flash-image"

VIDEO_MODEL = "gemini-omni-1.1-flash"


# ==================================================
# PERFORMANCE SETTINGS
# ==================================================

MAX_OUTPUT_TOKENS = 2048

CHAT_HISTORY_LIMIT = 4

VIDEO_PROCESSING_TIMEOUT = 300
VIDEO_POLL_INTERVAL = 3

MAX_RETRIES = 2
RETRY_BASE_DELAY = 1

IMAGE_TIMEOUT = 40

SYSTEM_INSTRUCTION = """
You are Alpha AI, an intelligent AI assistant.

1. Answer accurately.
2. Keep simple answers short.
3. Explain difficult questions clearly.
4. Use mathematical reasoning when needed.
5. Avoid unnecessary introductions.
6. Use simple English.
7. Never invent facts.
8. Answer the actual question.
9. Be helpful and professional.
"""


# ==================================================
# ERROR HANDLING
# ==================================================

def get_error_details(error):

    message = str(error)

    logger.error("API ERROR: %s", message)

    return message


def is_temporary_error(error):

    message = str(error).lower()

    terms = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "rate limit",
        "unavailable",
        "overloaded",
        "timeout",
        "temporarily unavailable",
        "internal server error"
    ]

    return any(term in message for term in terms)


def is_quota_error(error):

    message = str(error).lower()

    terms = [
        "quota exceeded",
        "daily limit",
        "billing",
        "limit: 0",
        "free_tier"
    ]

    return any(term in message for term in terms)


def error_message(error):

    message = get_error_details(error)

    if is_quota_error(error):

        return (
            "Alpha AI API quota reached. "
            "Please try again after the quota resets."
        )

    if is_temporary_error(error):

        return (
            "Alpha AI is experiencing high demand. "
            "Please try again shortly."
        )

    if "404" in message:

        return (
            "The requested AI model is unavailable "
            "for this API key."
        )

    if "403" in message:

        return (
            "API access was denied. "
            "Please check API permissions."
        )

    return (
        "Alpha AI could not complete your request. "
        "Please try again later."
    )


# ==================================================
# RETRY SYSTEM
# ==================================================

def retry_delay(attempt):

    return RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(
        0, 0.5
    )


def run_with_retries(function, retries=MAX_RETRIES):

    last_error = None

    for attempt in range(retries):

        try:

            return function()

        except Exception as error:

            last_error = error

            logger.warning(
                "Attempt %s failed: %s",
                attempt + 1,
                error
            )

            if (
                not is_temporary_error(error)
                or is_quota_error(error)
            ):

                raise

            if attempt < retries - 1:

                time.sleep(retry_delay(attempt))

    raise last_error


# ==================================================
# CHAT HISTORY
# ==================================================

def prepare_chat_contents(message, history):

    contents = []

    for item in (history or [])[-CHAT_HISTORY_LIMIT:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ["user", "assistant"]:
            continue

        if isinstance(content, str) and content.strip():

            contents.append(
                types.Content(
                    role="user" if role == "user" else "model",
                    parts=[
                        types.Part.from_text(text=content)
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

    return contents


# ==================================================
# 1. CHAT
# ==================================================

def alpha_chat(message, history):

    if not message or not message.strip():

        yield "Please enter a message."

        return

    start = time.monotonic()

    contents = prepare_chat_contents(
        message.strip(),
        history
    )

    for model in [CHAT_MODEL, BACKUP_MODEL]:

        answer = ""

        try:

            logger.info("CHAT MODEL: %s", model)

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )

            stream = run_with_retries(
                lambda: client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config
                )
            )

            for chunk in stream:

                if chunk.text:

                    answer += chunk.text

                    yield answer

            logger.info(
                "CHAT TIME: %.2f seconds",
                time.monotonic() - start
            )

            if not answer:

                yield "No response generated."

            return

        except Exception as error:

            logger.exception("CHAT ERROR")

            if answer:

                yield answer + "\n\nConnection interrupted."

                return

            if model == CHAT_MODEL and is_temporary_error(error):

                continue

            yield error_message(error)

            return


# ==================================================
# 2. IMAGE ANALYZER
# ==================================================

def analyze_image(image_path, question):

    if not image_path:

        return "Please upload an image."

    if not question or not question.strip():

        question = "Describe this image clearly."

    start = time.monotonic()

    try:

        with Image.open(image_path) as source:

            image = source.convert("RGB")

            image.thumbnail((1600, 1600))

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=85
            )

            image_bytes = buffer.getvalue()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    ),
                    types.Part.from_text(
                        text=question.strip()
                    )
                ]
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=2048
        )

        try:

            response = run_with_retries(
                lambda: client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=contents,
                    config=config
                )
            )

        except Exception as primary_error:

            if not is_temporary_error(primary_error):

                raise

            response = run_with_retries(
                lambda: client.models.generate_content(
                    model=BACKUP_MODEL,
                    contents=contents,
                    config=config
                )
            )

        answer = response.text or "No analysis generated."

        elapsed = time.monotonic() - start

        return (
            answer
            + f"\n\nAnalysis time: {elapsed:.2f} seconds."
        )

    except Exception as error:

        logger.exception("IMAGE ANALYSIS ERROR")

        return error_message(error)


# ==================================================
# 3. FAST IMAGE GENERATOR
# ==================================================

def generate_image(prompt):

    if not prompt or not prompt.strip():

        return None, "Please enter an image description."

    start = time.monotonic()

    try:

        logger.info("IMAGE GENERATION STARTED")

        # Use the updated image model.
        # Request one standard 1K image.

        interaction = run_with_retries(
            lambda: client.interactions.create(
                model=IMAGE_MODEL,
                input=prompt.strip(),
                response_format={
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "aspect_ratio": "1:1",
                    "image_size": "1K"
                },
                timeout=IMAGE_TIMEOUT
            ),
            retries=2
        )

        generated = getattr(
            interaction,
            "output_image",
            None
        )

        if generated is None:

            return (
                None,
                "The image model did not return an image."
            )

        image_data = getattr(
            generated,
            "data",
            None
        )

        if not image_data:

            return (
                None,
                "The generated image data was empty."
            )

        if isinstance(image_data, str):

            image_bytes = base64.b64decode(
                image_data,
                validate=True
            )

        else:

            image_bytes = image_data

        if not image_bytes:

            return None, "Invalid image data received."

        # Validate the generated image.

        with Image.open(
            io.BytesIO(image_bytes)
        ) as generated_image:

            generated_image.verify()

        # Load validated image.

        with Image.open(
            io.BytesIO(image_bytes)
        ) as generated_image:

            image = generated_image.convert("RGB")

        elapsed = time.monotonic() - start

        logger.info(
            "IMAGE GENERATION COMPLETED: %.2f seconds",
            elapsed
        )

        return (
            image,
            (
                "Image successfully generated! "
                f"Time: {elapsed:.2f} seconds."
            )
        )

    except Exception as error:

        logger.exception("IMAGE GENERATION ERROR")

        return None, error_message(error)


# ==================================================
# VIDEO FILE STATE
# ==================================================

def get_file_state(video_file):

    state = getattr(
        video_file,
        "state",
        None
    )

    if state is None:

        return ""

    return getattr(
        state,
        "name",
        str(state)
    ).upper()


# ==================================================
# 4. VIDEO ANALYZER
# ==================================================

def analyze_video(video, question):

    if not video:

        return "Please upload a video."

    if not question or not question.strip():

        question = "Explain what happens in this video."

    video_file = None

    start = time.monotonic()

    try:

        video_file = run_with_retries(
            lambda: client.files.upload(file=video)
        )

        processing_start = time.monotonic()

        while True:

            state = get_file_state(video_file)

            if state == "ACTIVE":

                break

            if state == "FAILED":

                return "Video processing failed."

            if (
                time.monotonic() - processing_start
                > VIDEO_PROCESSING_TIMEOUT
            ):

                return "Video processing timed out."

            time.sleep(VIDEO_POLL_INTERVAL)

            video_file = client.files.get(
                name=video_file.name
            )

        response = run_with_retries(
            lambda: client.models.generate_content(
                model=CHAT_MODEL,
                contents=[
                    video_file,
                    question
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=1024
                )
            )
        )

        elapsed = time.monotonic() - start

        return (
            (response.text or "No analysis generated.")
            + f"\n\nAnalysis time: {elapsed:.2f} seconds."
        )

    except Exception as error:

        logger.exception("VIDEO ANALYSIS ERROR")

        return error_message(error)

    finally:

        if video_file:

            try:

                client.files.delete(
                    name=video_file.name
                )

            except Exception:

                logger.warning("VIDEO CLEANUP FAILED")


# ==================================================
# 5. VIDEO GENERATOR
# ==================================================

def generate_video(prompt):

    if not prompt or not prompt.strip():

        return None, "Please enter a video description."

    start = time.monotonic()

    output_path = None

    try:

        interaction = run_with_retries(
            lambda: client.interactions.create(
                model=VIDEO_MODEL,
                input=prompt.strip(),
                response_format={
                    "type": "video",
                    "aspect_ratio": "16:9",
                    "resolution": "360p",
                    "delivery": "uri"
                }
            )
        )

        video_output = getattr(
            interaction,
            "output_video",
            None
        )

        if not video_output:

            return None, "No video response received."

        video_uri = getattr(
            video_output,
            "uri",
            None
        )

        if video_uri:

            file_id = (
                video_uri.split("/")[-1].split("?")[0]
            )

            file_name = f"files/{file_id}"

            processing_start = time.monotonic()

            while True:

                video_file = client.files.get(
                    name=file_name
                )

                state = get_file_state(video_file)

                if state == "ACTIVE":

                    break

                if state == "FAILED":

                    return None, "Video generation failed."

                if (
                    time.monotonic() - processing_start
                    > VIDEO_PROCESSING_TIMEOUT
                ):

                    return None, "Video processing timed out."

                time.sleep(VIDEO_POLL_INTERVAL)

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as output_file:

                output_path = output_file.name

            client.files.download(
                file=video_uri,
                destination=output_path
            )

        else:

            video_data = getattr(
                video_output,
                "data",
                None
            )

            if not video_data:

                return None, "Video data was not returned."

            video_bytes = base64.b64decode(video_data)

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as output_file:

                output_file.write(video_bytes)

                output_path = output_file.name

        elapsed = time.monotonic() - start

        return (
            output_path,
            f"Video generated in {elapsed:.2f} seconds."
        )

    except Exception as error:

        logger.exception("VIDEO GENERATION ERROR")

        if output_path and os.path.exists(output_path):

            try:

                os.remove(output_path)

            except OSError:

                pass

        return None, error_message(error)


# ==================================================
# 6. ALPHA AI INTERFACE
# ==================================================

with gr.Blocks(
    title="ALPHA AI"
) as app:

    gr.Markdown(
        """
        # 🚀 ALPHA AI

        ### Think Beyond Limits

        Your intelligent AI assistant.

        Chat | Image Analysis | Image Creation |
        Video Analysis | Video Generation
        """
    )

    # CHAT

    with gr.Tab("Chat"):

        gr.ChatInterface(
            fn=alpha_chat,
            title="Alpha AI Chat",
            description="Ask Alpha AI anything."
        )

    # IMAGE ANALYZER

    with gr.Tab("Image Analyzer"):

        gr.Markdown(
            "### Upload an image and ask Alpha AI."
        )

        image_input = gr.Image(
            label="Upload or Take a Photo",
            type="filepath",
            sources=["upload", "webcam"]
        )

        image_question = gr.Textbox(
            label="Ask Alpha AI",
            placeholder="Describe this image..."
        )

        image_analyze_button = gr.Button(
            "Analyze Image",
            variant="primary"
        )

        image_analysis_output = gr.Markdown()

        image_analyze_button.click(
            fn=analyze_image,
            inputs=[
                image_input,
                image_question
            ],
            outputs=image_analysis_output
        )

    # IMAGE CREATOR

    with gr.Tab("Image Creator"):

        gr.Markdown(
            """
            ### 🎨 Alpha AI Image Creator

            Create images using AI.
            """
        )

        image_prompt = gr.Textbox(
            label="Image Description",
            placeholder="A futuristic city on Mars...",
            lines=3
        )

        image_button = gr.Button(
            "Generate Image",
            variant="primary"
        )

        image_output = gr.Image(
            label="Generated Image",
            type="pil"
        )

        image_status = gr.Markdown()

        image_button.click(
            fn=generate_image,
            inputs=image_prompt,
            outputs=[
                image_output,
                image_status
            ]
        )

    # VIDEO ANALYZER

    with gr.Tab("Video Analyzer"):

        gr.Markdown(
            "### Analyze Videos Using Alpha AI"
        )

        video_input = gr.Video(
            label="Upload Video",
            sources=["upload"]
        )

        video_question = gr.Textbox(
            label="Your Question",
            placeholder="Explain this video..."
        )

        analyze_button = gr.Button(
            "Analyze Video",
            variant="primary"
        )

        analysis_output = gr.Textbox(
            label="Video Analysis",
            lines=8
        )

        analyze_button.click(
            fn=analyze_video,
            inputs=[
                video_input,
                video_question
            ],
            outputs=analysis_output
        )

    # VIDEO GENERATOR

    with gr.Tab("Video Generator"):

        gr.Markdown(
            "### Generate Videos Using Alpha AI"
        )

        video_prompt = gr.Textbox(
            label="Video Description",
            placeholder="Saturn rotating in space...",
            lines=3
        )

        generate_video_button = gr.Button(
            "Generate Video",
            variant="primary"
        )

        generated_video = gr.Video(
            label="Generated Video"
        )

        video_status = gr.Markdown()

        generate_video_button.click(
            fn=generate_video,
            inputs=video_prompt,
            outputs=[
                generated_video,
                video_status
            ]
        )


# ==================================================
# 7. QUEUE
# ==================================================

app.queue(
    max_size=10,
    default_concurrency_limit=2
)


# ==================================================
# 8. SERVER STARTUP
# ==================================================

if __name__ == "__main__":

    PORT = int(
        os.environ.get("PORT", "10000")
    )

    logger.info(
        "STARTING ALPHA AI 3.7 ON PORT %s",
        PORT
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        theme=gr.themes.Soft(),
        show_error=False
)
