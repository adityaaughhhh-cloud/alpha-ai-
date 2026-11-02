import os
import io
import time
import base64
import tempfile
import logging

import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# ==================================================
# ALPHA AI 3.4 - STABILITY & SPEED EDITION
# ==================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ALPHA")


# ==================================================
# API CONFIGURATION
# ==================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing in Render environment variables."
    )

client = genai.Client(api_key=API_KEY)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

CHAT_MODEL = "gemini-3.5-flash-lite"

BACKUP_MODEL = "gemini-3.1-flash-lite"

IMAGE_MODEL = "gemini-3.1-flash-image"

VIDEO_MODEL = "gemini-omni-1.1-flash"


# ==================================================
# PERFORMANCE SETTINGS
# ==================================================

max_output_tokens = 10000

CHAT_HISTORY_LIMIT = 4

VIDEO_PROCESSING_TIMEOUT = 300

VIDEO_POLL_INTERVAL = 3

SYSTEM_INSTRUCTION = """
You are Alpha AI, an intelligent AI assistant.

Instructions:

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

def is_temporary_error(error):

    message = str(error).lower()

    temporary_errors = [
        "503",
        "unavailable",
        "overloaded",
        "429",
        "resource_exhausted",
        "rate limit",
        "500",
        "internal server error",
        "timeout"
    ]

    return any(
        item in message
        for item in temporary_errors
    )


def error_message(error):

    logger.exception("ALPHA ERROR: %s", error)

    if is_temporary_error(error):

        return (
            "Alpha AI is experiencing high demand. "
            "Please try again shortly."
        )

    return (
        "Alpha AI could not complete your request. "
        "Please try again."
    )


# ==================================================
# CHAT HISTORY PROCESSING
# ==================================================

def prepare_chat_contents(message, history):

    contents = []

    for item in (history or [])[-CHAT_HISTORY_LIMIT:]:

        role = None
        content = None

        # Gradio messages format

        if isinstance(item, dict):

            role = item.get("role")

            content = item.get("content")

        # Older Gradio tuple format

        elif isinstance(item, (list, tuple)) and len(item) == 2:

            user_message, assistant_message = item

            if isinstance(user_message, str):

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=user_message
                            )
                        ]
                    )
                )

            if isinstance(assistant_message, str):

                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=assistant_message
                            )
                        ]
                    )
                )

            continue

        if role not in ["user", "assistant"]:
            continue

        if isinstance(content, str) and content.strip():

            contents.append(
                types.Content(
                    role="user" if role == "user" else "model",
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


# ==================================================
# 1. ALPHA AI FAST CHAT
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

    models = [
        CHAT_MODEL,
        BACKUP_MODEL
    ]

    for index, model in enumerate(models):

        answer = ""

        try:

            logger.info(
                "CHAT MODEL: %s",
                model
            )

            config = types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                temperature=0.3,

                max_output_tokens=MAX_OUTPUT_TOKENS,

                candidate_count=1
            )

            stream = client.models.generate_content_stream(

                model=model,

                contents=contents,

                config=config
            )

            first_token = None

            for chunk in stream:

                if chunk.text:

                    if first_token is None:

                        first_token = time.monotonic()

                        logger.info(
                            "FIRST TOKEN TIME: %.3f seconds",
                            first_token - start
                        )

                    answer += chunk.text

                    yield answer

            elapsed = time.monotonic() - start

            logger.info(
                "CHAT TOTAL TIME: %.3f seconds",
                elapsed
            )

            if not answer:

                yield "No response generated."

            return

        except Exception as error:

            logger.exception(
                "CHAT MODEL FAILED: %s",
                model
            )

            if answer:

                yield (
                    answer
                    + "\n\nConnection interrupted."
                )

                return

            if (
                index == 0
                and is_temporary_error(error)
            ):

                logger.warning(
                    "SWITCHING TO BACKUP MODEL"
                )

                continue

            yield error_message(error)

            return


# ==================================================
# 2. IMAGE GENERATOR
# ==================================================

def generate_image(prompt):

    if not prompt or not prompt.strip():

        return None, "Please enter an image description."

    start = time.monotonic()

    try:

        logger.info(
            "IMAGE GENERATION STARTED"
        )

        interaction = client.interactions.create(

            model=IMAGE_MODEL,

            input=prompt.strip(),

            response_format={
                "type": "image",
                "mime_type": "image/jpeg",
                "aspect_ratio": "1:1",
                "image_size": "1K"
            }
        )

        # Extract image response

        generated = getattr(
            interaction,
            "output_image",
            None
        )

        if not generated:

            return None, "No image was returned by the model."

        image_data = getattr(
            generated,
            "data",
            None
        )

        if not image_data:

            return None, "Image data is empty."

        if isinstance(image_data, str):

            image_bytes = base64.b64decode(
                image_data
            )

        else:

            image_bytes = image_data

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        image = image.convert("RGB")

        elapsed = time.monotonic() - start

        logger.info(
            "IMAGE GENERATION TIME: %.2f seconds",
            elapsed
        )

        return (
            image,
            f"Image successfully generated in {elapsed:.2f} seconds."
        )

    except Exception as error:

        logger.exception(
            "IMAGE GENERATION FAILED"
        )

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
# 3. VIDEO ANALYZER
# ==================================================

def analyze_video(video, question):

    if not video:

        return "Please upload a video."

    if not question or not question.strip():

        question = (
            "Explain what happens in this video."
        )

    video_file = None

    start = time.monotonic()

    try:

        logger.info(
            "VIDEO ANALYSIS STARTED"
        )

        video_file = client.files.upload(
            file=video
        )

        processing_start = time.monotonic()

        while True:

            state = get_file_state(
                video_file
            )

            if state == "ACTIVE":

                break

            if state == "FAILED":

                return "Video processing failed."

            if (
                time.monotonic() - processing_start
                > VIDEO_PROCESSING_TIMEOUT
            ):

                return "Video processing timed out."

            time.sleep(
                VIDEO_POLL_INTERVAL
            )

            video_file = client.files.get(
                name=video_file.name
            )

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=[
                video_file,
                question
            ],

            config=types.GenerateContentConfig(

                max_output_tokens=1024
            )
        )

        elapsed = time.monotonic() - start

        logger.info(
            "VIDEO ANALYSIS TIME: %.2f seconds",
            elapsed
        )

        return (
            response.text
            or "No analysis generated."
        )

    except Exception as error:

        logger.exception(
            "VIDEO ANALYSIS FAILED"
        )

        return error_message(error)

    finally:

        if video_file:

            try:

                client.files.delete(
                    name=video_file.name
                )

            except Exception:

                logger.warning(
                    "VIDEO FILE CLEANUP FAILED"
                )


# ==================================================
# 4. VIDEO GENERATOR
# ==================================================

def generate_video(prompt):

    if not prompt or not prompt.strip():

        return None, "Please enter a video description."

    start = time.monotonic()

    output_path = None

    try:

        logger.info(
            "VIDEO GENERATION STARTED"
        )

        interaction = client.interactions.create(

            model=VIDEO_MODEL,

            input=prompt.strip(),

            response_format={
                "type": "video",
                "aspect_ratio": "16:9",
                "resolution": "360p",
                "delivery": "uri"
            }
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

            file_id = video_uri.split("/")[-1].split("?")[0]

            file_name = f"files/{file_id}"

            processing_start = time.monotonic()

            while True:

                video_file = client.files.get(
                    name=file_name
                )

                state = get_file_state(
                    video_file
                )

                if state == "ACTIVE":

                    break

                if state == "FAILED":

                    return None, "Video generation failed."

                if (
                    time.monotonic() - processing_start
                    > VIDEO_PROCESSING_TIMEOUT
                ):

                    return None, "Video processing timed out."

                time.sleep(3)

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

            video_bytes = base64.b64decode(
                video_data
            )

            with tempfile.NamedTemporaryFile(

                suffix=".mp4",

                delete=False

            ) as output_file:

                output_file.write(
                    video_bytes
                )

                output_path = output_file.name

        elapsed = time.monotonic() - start

        logger.info(
            "VIDEO GENERATION TIME: %.2f seconds",
            elapsed
        )

        return (
            output_path,
            f"Video generated in {elapsed:.2f} seconds."
        )

    except Exception as error:

        logger.exception(
            "VIDEO GENERATION FAILED"
        )

        if output_path and os.path.exists(output_path):

            try:
                os.remove(output_path)
            except OSError:
                pass

        return None, error_message(error)


# ==================================================
# 5. ALPHA AI INTERFACE
# ==================================================

with gr.Blocks(
    title="ALPHA AI"
) as app:

    gr.Markdown(
        """
        # 🚀 ALPHA AI

        ### Think Beyond Limits

        Your intelligent AI assistant.

        Fast Chat | Image Creation |
        Video Analysis | Video Generation
        """
    )

    # ----------------------------------------------
    # CHAT
    # ----------------------------------------------

    with gr.Tab("Chat"):

        gr.ChatInterface(

            fn=alpha_chat,

            title="Alpha AI Chat",

            description=(
                "Ask Alpha AI anything."
            )
        )

    # ----------------------------------------------
    # IMAGE GENERATOR
    # ----------------------------------------------

    with gr.Tab("Image Creator"):

        gr.Markdown(
            "### Create images using Alpha AI"
        )

        image_prompt = gr.Textbox(

            label="Image Description",

            placeholder=(
                "A futuristic city on Mars..."
            ),

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

    # ----------------------------------------------
    # VIDEO ANALYZER
    # ----------------------------------------------

    with gr.Tab("Video Analyzer"):

        gr.Markdown(
            "### Analyze videos using Alpha AI"
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

    # ----------------------------------------------
    # VIDEO GENERATOR
    # ----------------------------------------------

    with gr.Tab("Video Generator"):

        gr.Markdown(
            "### Generate videos using Alpha AI"
        )

        video_prompt = gr.Textbox(

            label="Video Description",

            placeholder=(
                "Saturn rotating in space, "
                "cinematic camera movement..."
            ),

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

            inputs=[
                video_prompt
            ],

            outputs=[
                generated_video,
                video_status
            ]
        )


# ==================================================
# 6. QUEUE CONFIGURATION
# ==================================================

app.queue(
    max_size=10,
    default_concurrency_limit=2
)


# ==================================================
# 7. SERVER STARTUP
# ==================================================

if __name__ == "__main__":

    PORT = int(
        os.environ.get("PORT", "10000")
    )

    logger.info(
        "STARTING ALPHA AI ON PORT %s",
        PORT
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        theme=gr.themes.Soft(),

        show_error=False
    )
