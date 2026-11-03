import os
import io
import time
import base64
import tempfile
import logging
import sqlite3
import threading

from datetime import date

import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# =====================================================
# ALPHA AI 4.1
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALPHA")


# =====================================================
# CONFIGURATION
# =====================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from Render environment variables."
    )

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gemini-3.5-flash-lite"
)

BACKUP_MODEL = os.environ.get(
    "BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)

IMAGE_MODEL = os.environ.get(
    "IMAGE_MODEL",
    "gemini-3.1-flash-image"
)

VIDEO_MODEL = os.environ.get(
    "VIDEO_MODEL",
    "gemini-omni-1.1-flash"
)

MAX_OUTPUT_TOKENS = 1024
CHAT_HISTORY_LIMIT = 6
IMAGE_DAILY_LIMIT = 20

VIDEO_TIMEOUT = 600
VIDEO_POLL_INTERVAL = 5

DB_PATH = os.environ.get(
    "DB_PATH",
    "alpha_usage.db"
)

PORT = int(os.environ.get("PORT", "10000"))

client = genai.Client(api_key=API_KEY)


# =====================================================
# SYSTEM INSTRUCTION
# =====================================================

SYSTEM_INSTRUCTION = """
You are Alpha AI, an intelligent AI assistant.

Identity:
Name: Alpha AI
Tagline: Think Beyond Limits.

Rules:

1. Give accurate and useful answers.
2. Never invent facts.
3. Answer the actual question.
4. Keep simple answers short.
5. Explain complicated topics clearly.
6. Use simple English.
7. Use mathematical reasoning when required.
8. Admit uncertainty when necessary.
9. Avoid unnecessary introductions.
10. Be professional and helpful.
11. Use conversation history when relevant.
12. Do not repeat information unnecessarily.

Your goal is to provide fast, useful and reliable answers.
"""


# =====================================================
# DATABASE
# =====================================================

db_lock = threading.Lock()


def get_connection():

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30
    )

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return connection


def initialize_database():

    with db_lock:

        connection = get_connection()

        try:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS image_usage (
                    session_id TEXT NOT NULL,
                    usage_date TEXT NOT NULL,
                    image_count INTEGER DEFAULT 0,
                    PRIMARY KEY (session_id, usage_date)
                )
            """)

            connection.commit()

        finally:
            connection.close()


initialize_database()


def get_image_usage(session_id):

    today = date.today().isoformat()

    with db_lock:

        connection = get_connection()

        try:

            result = connection.execute(
                """
                SELECT image_count
                FROM image_usage
                WHERE session_id = ?
                AND usage_date = ?
                """,
                (session_id, today)
            ).fetchone()

            return result[0] if result else 0

        finally:
            connection.close()


def consume_image_credit(session_id):

    today = date.today().isoformat()

    with db_lock:

        connection = get_connection()

        try:

            connection.execute(
                """
                INSERT OR IGNORE INTO image_usage
                (session_id, usage_date, image_count)
                VALUES (?, ?, 0)
                """,
                (session_id, today)
            )

            cursor = connection.execute(
                """
                UPDATE image_usage
                SET image_count = image_count + 1
                WHERE session_id = ?
                AND usage_date = ?
                AND image_count < ?
                """,
                (
                    session_id,
                    today,
                    IMAGE_DAILY_LIMIT
                )
            )

            connection.commit()

            return cursor.rowcount == 1

        finally:
            connection.close()


# =====================================================
# ERROR HANDLING
# =====================================================

def get_error(error):

    logger.exception("ALPHA ERROR")

    message = str(error).lower()

    if "429" in message or "quota" in message:
        return "API quota reached. Please try again later."

    if "503" in message or "overloaded" in message:
        return "Alpha AI is experiencing high demand. Try again shortly."

    if "404" in message:
        return "The selected AI model is unavailable. Check model configuration."

    if "403" in message:
        return "API access denied. Check Google API permissions."

    if "timeout" in message:
        return "The request took too long. Please try again."

    if "api key" in message:
        return "There is a problem with the Gemini API key."

    return "Alpha AI could not complete your request. Please try again."


# =====================================================
# CHAT HISTORY
# =====================================================

def prepare_history(history):

    contents = []

    for item in (history or [])[-CHAT_HISTORY_LIMIT:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ["user", "assistant"]:
            continue

        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

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

    return contents


# =====================================================
# 1. AI CHAT
# =====================================================

def alpha_chat(message, history):

    if not message or not message.strip():
        yield "Please enter a message."
        return

    start = time.monotonic()

    contents = prepare_history(history)

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=message.strip()
                )
            ]
        )
    )

    models = [
        CHAT_MODEL,
        BACKUP_MODEL
    ]

    for index, model in enumerate(models):

        answer = ""

        try:

            logger.info("CHAT MODEL: %s", model)

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )

            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config
            )

            for chunk in stream:

                if chunk.text:

                    answer += chunk.text

                    yield answer

            if answer:

                elapsed = time.monotonic() - start

                logger.info(
                    "CHAT COMPLETED: %.2f seconds",
                    elapsed
                )

                return

            yield "No response generated."
            return

        except Exception as error:

            logger.exception("CHAT ERROR")

            if answer:
                yield answer + "\n\nConnection interrupted."
                return

            if index == 0:
                logger.info("Trying backup model.")
                continue

            yield get_error(error)
            return


# =====================================================
# 2. IMAGE ANALYSIS
# =====================================================

def analyze_image(image_path, question):

    if not image_path:
        return "Please upload an image."

    if not question or not question.strip():
        question = "Describe this image clearly."

    start = time.monotonic()

    try:

        with Image.open(image_path) as source:

            image = source.convert("RGB")

            image.thumbnail((1400, 1400))

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=82
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
            max_output_tokens=1024
        )

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=config
        )

        answer = response.text or "No analysis generated."

        elapsed = time.monotonic() - start

        return (
            answer
            + f"\n\nAnalysis time: {elapsed:.2f} seconds."
        )

    except Exception as error:
        return get_error(error)


# =====================================================
# 3. IMAGE GENERATION
# =====================================================

def generate_image(prompt, request: gr.Request):

    if not prompt or not prompt.strip():
        return None, "Enter an image description."

    session_id = (
        request.session_hash
        if request
        else "unknown-session"
    )

    if get_image_usage(session_id) >= IMAGE_DAILY_LIMIT:
        return None, "Daily image limit reached."

    start = time.monotonic()

    try:

        logger.info("IMAGE GENERATION STARTED")

        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt.strip(),
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )

        generated_image = None

        for part in (response.parts or []):

            if getattr(part, "inline_data", None):

                generated_image = part.as_image()
                break

        if generated_image is None:
            return None, "The image model returned no image."

        if not consume_image_credit(session_id):
            return None, "Daily image limit reached."

        used = get_image_usage(session_id)

        elapsed = time.monotonic() - start

        logger.info(
            "IMAGE GENERATED IN %.2f SECONDS",
            elapsed
        )

        return (
            generated_image,
            (
                "Image generated successfully!\n\n"
                f"Time: {elapsed:.2f} seconds.\n"
                f"Daily usage: {used}/{IMAGE_DAILY_LIMIT}"
            )
        )

    except Exception as error:
        return None, get_error(error)


# =====================================================
# VIDEO FILE STATE
# =====================================================

def get_file_state(video_file):

    state = getattr(
        video_file,
        "state",
        None
    )

    return getattr(
        state,
        "name",
        str(state)
    ).upper()


# =====================================================
# 4. VIDEO ANALYSIS
# =====================================================

def analyze_video(video_path, question):

    if not video_path:
        return "Please upload a video."

    if not question or not question.strip():
        question = "Explain what happens in this video."

    video_file = None

    start = time.monotonic()

    try:

        logger.info("VIDEO UPLOAD STARTED")

        video_file = client.files.upload(
            file=video_path
        )

        processing_start = time.monotonic()

        while True:

            video_file = client.files.get(
                name=video_file.name
            )

            state = get_file_state(video_file)

            if state == "ACTIVE":
                break

            if state == "FAILED":
                return "Video processing failed."

            if time.monotonic() - processing_start > VIDEO_TIMEOUT:
                return "Video processing timed out."

            time.sleep(3)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri,
                        mime_type=video_file.mime_type
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
            max_output_tokens=1024
        )

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=config
        )

        answer = response.text or "No video analysis generated."

        elapsed = time.monotonic() - start

        return (
            answer
            + f"\n\nAnalysis time: {elapsed:.2f} seconds."
        )

    except Exception as error:
        return get_error(error)

    finally:

        if video_file:

            try:

                client.files.delete(
                    name=video_file.name
                )

            except Exception:

                logger.warning("VIDEO CLEANUP FAILED")


# =====================================================
# 5. VIDEO GENERATION
# =====================================================

def generate_video(prompt):

    if not prompt or not prompt.strip():
        return None, "Enter a video description."

    output_path = None

    start = time.monotonic()

    try:

        logger.info("VIDEO GENERATION STARTED")

        interaction = client.interactions.create(
            model=VIDEO_MODEL,
            input=prompt.strip(),
            response_format={
                "type": "video",
                "aspect_ratio": "16:9",
                "resolution": "360p"
            }
        )

        video_output = getattr(
            interaction,
            "output_video",
            None
        )

        if video_output is None:
            return None, "The video model returned no video."

        video_data = getattr(
            video_output,
            "data",
            None
        )

        video_uri = getattr(
            video_output,
            "uri",
            None
        )

        if video_data:

            video_bytes = base64.b64decode(video_data)

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as output:

                output.write(video_bytes)
                output_path = output.name

        elif video_uri:

            file_name = video_uri.split("/")[-1]
            file_name = file_name.split("?")[0]

            processing_start = time.monotonic()

            while True:

                video_file = client.files.get(
                    name=f"files/{file_name}"
                )

                state = get_file_state(video_file)

                if state == "ACTIVE":
                    break

                if state == "FAILED":
                    return None, "Video generation failed."

                if time.monotonic() - processing_start > VIDEO_TIMEOUT:
                    return None, "Video processing timed out."

                time.sleep(VIDEO_POLL_INTERVAL)

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as output:

                output_path = output.name

            client.files.download(
                file=video_uri,
                destination=output_path
            )

        else:
            return None, "No video data was returned."

        if not output_path or not os.path.exists(output_path):
            return None, "Generated video could not be saved."

        if os.path.getsize(output_path) == 0:
            return None, "Generated video file is empty."

        elapsed = time.monotonic() - start

        logger.info(
            "VIDEO GENERATED IN %.2f SECONDS",
            elapsed
        )

        return (
            output_path,
            (
                "360p video generated successfully!\n\n"
                f"Time: {elapsed:.2f} seconds."
            )
        )

    except Exception as error:

        logger.exception("VIDEO GENERATION ERROR")

        if output_path and os.path.exists(output_path):

            try:
                os.remove(output_path)

            except OSError:
                pass

        return None, get_error(error)


# =====================================================
# 6. DESIGN
# =====================================================

CSS = """

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

#alpha-header {
    text-align: center;
    padding: 18px;
}

#alpha-header h1 {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 2px;
}

#alpha-header p {
    font-size: 14px;
    opacity: 0.7;
}

#chatbot {
    border-radius: 16px;
}

button {
    border-radius: 12px !important;
}

textarea {
    border-radius: 14px !important;
}

"""


# =====================================================
# 7. USER INTERFACE
# GRADIO 6.31.0
# =====================================================

with gr.Blocks(
    title="ALPHA AI"
) as app:

    gr.HTML(
        """
        <div id="alpha-header">
            <h1>🚀 ALPHA AI</h1>
            <p>THINK BEYOND LIMITS</p>
            <p>Your intelligent AI assistant.</p>
        </div>
        """
    )

    with gr.Tabs():

        # CHAT

        with gr.Tab("💬 Chat"):

            gr.ChatInterface(
                fn=alpha_chat,
                title="Alpha AI",
                description="Ask anything. Get fast and intelligent answers.",
                chatbot=gr.Chatbot(
                    height=530
                ),
                textbox=gr.Textbox(
                    placeholder="Message Alpha AI...",
                    container=True,
                    scale=7,
                    submit_btn="Send",
                    stop_btn="Stop"
                )
            )

        # IMAGE ANALYSIS

        with gr.Tab("🖼️ Image Analysis"):

            gr.Markdown("## Understand any image")

            image_input = gr.Image(
                label="Upload Image",
                type="filepath"
            )

            image_question = gr.Textbox(
                label="Ask Alpha",
                placeholder="What is in this image?"
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

        with gr.Tab("🎨 Image Creator"):

            gr.Markdown(
                """
                ## Alpha AI Image Creator

                Create images using artificial intelligence.

                Free daily allowance: 20 images per session.
                """
            )

            image_prompt = gr.Textbox(
                label="Describe your image",
                placeholder="A futuristic city on Mars with flying vehicles...",
                lines=3
            )

            image_button = gr.Button(
                "Generate Image",
                variant="primary"
            )

            generated_image = gr.Image(
                label="Your Generated Image",
                type="pil"
            )

            image_status = gr.Markdown()

            image_button.click(
                fn=generate_image,
                inputs=image_prompt,
                outputs=[
                    generated_image,
                    image_status
                ]
            )

        # VIDEO ANALYSIS

        with gr.Tab("🎬 Video Analysis"):

            gr.Markdown("## Understand Videos")

            video_input = gr.Video(
                label="Upload Video",
                sources=["upload"]
            )

            video_question = gr.Textbox(
                label="Ask Alpha",
                placeholder="Explain what happens in this video."
            )

            video_analyze_button = gr.Button(
                "Analyze Video",
                variant="primary"
            )

            video_analysis_output = gr.Markdown()

            video_analyze_button.click(
                fn=analyze_video,
                inputs=[
                    video_input,
                    video_question
                ],
                outputs=video_analysis_output
            )

        # VIDEO CREATOR

        with gr.Tab("🎥 Video Creator"):

            gr.Markdown(
                """
                ## Alpha AI Video Creator

                Generate videos from text descriptions.

                Resolution: 360p
                """
            )

            video_prompt = gr.Textbox(
                label="Describe your video",
                placeholder="A cinematic view of Saturn rotating in deep space...",
                lines=3
            )

            video_generate_button = gr.Button(
                "Generate Video",
                variant="primary"
            )

            generated_video = gr.Video(
                label="Generated Video"
            )

            video_status = gr.Markdown()

            video_generate_button.click(
                fn=generate_video,
                inputs=video_prompt,
                outputs=[
                    generated_video,
                    video_status
                ]
            )

    gr.Markdown(
        """
        ---

        **ALPHA AI** | Think Beyond Limits

        Powered by Google Gemini.
        """
    )


# =====================================================
# 8. PERFORMANCE QUEUE
# =====================================================

app.queue(
    max_size=20,
    default_concurrency_limit=2
)


# =====================================================
# 9. START SERVER
# =====================================================

if __name__ == "__main__":

    logger.info("STARTING ALPHA AI 4.1")

    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        ),
        css=CSS,
        show_error=True
        )
