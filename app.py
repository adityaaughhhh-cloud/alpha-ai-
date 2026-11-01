import os
import io
import time
import base64
import tempfile
import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# ==========================================
# ALPHA AI 3.2 - SPEED EDITION
# ==========================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in Render.")

client = genai.Client(api_key=API_KEY)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

CHAT_MODEL = "gemini-3.5-flash-lite"

BACKUP_MODEL = "gemini-3.1-flash-lite"

IMAGE_MODEL = "gemini-3.1-flash-lite-image"

VIDEO_MODEL = "gemini-omni-1.1-flash"


# ==========================================
# SPEED CONFIGURATION
# ==========================================

MAX_OUTPUT_TOKENS = 128

CHAT_HISTORY_LIMIT = 4

SYSTEM_INSTRUCTION = """
You are Alpha AI, an intelligent AI assistant.

Rules:
1. Answer accurately.
2. Answer simple questions briefly.
3. Explain difficult questions clearly.
4. Use mathematical reasoning when needed.
5. Avoid unnecessary introductions.
6. Use simple and clear language.
7. Never invent facts.
8. Focus on the user's actual question.
9. Keep simple answers short.
"""


# ==========================================
# ERROR DETECTION
# ==========================================

def is_temporary_error(error):

    message = str(error).lower()

    errors = [
        "503",
        "unavailable",
        "overloaded",
        "429",
        "resource_exhausted",
        "rate limit",
        "500",
        "internal server error"
    ]

    return any(
        item in message
        for item in errors
    )


# ==========================================
# CHAT HISTORY OPTIMIZATION
# ==========================================

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


# ==========================================
# 1. ALPHA AI ULTRA FAST CHAT
# ==========================================

def alpha_chat(message, history):

    if not message or not message.strip():

        yield "Please enter a message."

        return

    start = time.monotonic()

    first_token_time = None

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

            print(
                f"ALPHA MODEL: {model}"
            )

            config = types.GenerateContentConfig(

                system_instruction=SYSTEM_INSTRUCTION,

                temperature=0.3,

                max_output_tokens=MAX_OUTPUT_TOKENS,

                candidate_count=1,

                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                )
            )

            # STREAMING GENERATION

            stream = client.models.generate_content_stream(

                model=model,

                contents=contents,

                config=config
            )

            for chunk in stream:

                if chunk.text:

                    if first_token_time is None:

                        first_token_time = (
                            time.monotonic() - start
                        )

                        print(
                            f"FIRST TOKEN: {first_token_time:.2f}s"
                        )

                    answer += chunk.text

                    yield answer

            total_time = (
                time.monotonic() - start
            )

            print(
                f"TOTAL RESPONSE TIME: {total_time:.2f}s"
            )

            if not answer:

                yield "No response generated."

            return

        except Exception as error:

            print(
                f"CHAT ERROR ({model}): {repr(error)}"
            )

            if answer:

                yield (
                    answer
                    + "\n\nConnection interrupted. "
                    "Please try again."
                )

                return

            if (
                index == 0
                and is_temporary_error(error)
            ):

                print(
                    "SWITCHING TO BACKUP MODEL"
                )

                continue

            yield (
                "Alpha AI is temporarily unavailable. "
                "Please try again shortly."
            )

            return


# ==========================================
# 2. IMAGE GENERATOR
# ==========================================

def generate_image(prompt):

    if not prompt or not prompt.strip():

        return None, "Enter an image description."

    start = time.monotonic()

    try:

        print(
            "ALPHA IMAGE GENERATION STARTED"
        )

        result = client.interactions.create(

            model=IMAGE_MODEL,

            input=prompt.strip(),

            response_format={
                "type": "image",
                "mime_type": "image/jpeg",
                "aspect_ratio": "1:1",
                "image_size": "1K"
            }
        )

        generated = result.output_image

        if not generated or not generated.data:

            return None, "No image was returned."

        image_bytes = base64.b64decode(
            generated.data
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        image = image.convert("RGB")

        elapsed = (
            time.monotonic() - start
        )

        print(
            f"IMAGE TIME: {elapsed:.2f}s"
        )

        return (
            image,
            f"Image generated in {elapsed:.2f} seconds!"
        )

    except Exception as error:

        print(
            "IMAGE ERROR:",
            repr(error)
        )

        return None, f"Image error: {str(error)}"


# ==========================================
# 3. VIDEO ANALYZER
# ==========================================

def analyze_video(video, question):

    try:

        if not video:

            return "Please upload a video."

        if not question or not question.strip():

            question = (
                "Explain what happens in this video."
            )

        start = time.monotonic()

        video_file = client.files.upload(
            file=video
        )

        processing_start = time.monotonic()

        while (
            not video_file.state
            or video_file.state.name == "PROCESSING"
        ):

            if (
                time.monotonic() - processing_start > 300
            ):

                return "Video processing timed out."

            time.sleep(3)

            video_file = client.files.get(
                name=video_file.name
            )

        if video_file.state.name != "ACTIVE":

            return "Video processing failed."

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

        elapsed = (
            time.monotonic() - start
        )

        print(
            f"VIDEO ANALYSIS TIME: {elapsed:.2f}s"
        )

        return (
            response.text
            or "No analysis generated."
        )

    except Exception as error:

        print(
            "VIDEO ANALYSIS ERROR:",
            repr(error)
        )

        return f"Video analysis error: {str(error)}"


# ==========================================
# 4. VIDEO GENERATOR
# ==========================================

def generate_video(prompt):

    start = time.monotonic()

    try:

        if not prompt or not prompt.strip():

            return None, "Enter a video description."

        print(
            "ALPHA VIDEO GENERATION STARTED"
        )

        interaction = client.interactions.create(

            model=VIDEO_MODEL,

            input=(
                "Create a short cinematic video. "
                + prompt.strip()
            ),

            response_format={
                "type": "video",
                "aspect_ratio": "16:9",
                "resolution": "360p"
            }
        )

        video_output = interaction.output_video

        if not video_output or not video_output.data:

            return None, "No video data returned."

        video_bytes = base64.b64decode(
            video_output.data
        )

        with tempfile.NamedTemporaryFile(

            suffix=".mp4",

            delete=False

        ) as output_file:

            output_file.write(
                video_bytes
            )

            output_path = output_file.name

        elapsed = (
            time.monotonic() - start
        )

        print(
            f"VIDEO GENERATION TIME: {elapsed:.2f}s"
        )

        return (
            output_path,
            f"Video generated in {elapsed:.2f} seconds!"
        )

    except Exception as error:

        print(
            "VIDEO GENERATION ERROR:",
            repr(error)
        )

        return None, f"Video generation error: {str(error)}"


# ==========================================
# 5. ALPHA AI INTERFACE
# ==========================================

with gr.Blocks(
    title="ALPHA AI"
) as app:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits

        Your AI assistant for chat, image creation,
        video analysis and video generation.
        """
    )

    # ======================================
    # CHAT
    # ======================================

    with gr.Tab("Chat"):

        gr.ChatInterface(

            fn=alpha_chat,

            title="Alpha AI Chatbot",

            description=(
                "Ask Alpha AI anything. "
                "Powered by fast Gemini models."
            )
        )

    # ======================================
    # IMAGE CREATOR
    # ======================================

    with gr.Tab("Image Creator"):

        gr.Markdown(
            "### Create images using Alpha AI"
        )

        image_prompt = gr.Textbox(

            label="Image Description",

            placeholder="A futuristic city on Mars..."
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

    # ======================================
    # VIDEO ANALYZER
    # ======================================

    with gr.Tab("Video Analyzer"):

        gr.Markdown(
            "### Upload a video and ask Alpha AI"
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

    # ======================================
    # VIDEO GENERATOR
    # ======================================

    with gr.Tab("Video Generator"):

        gr.Markdown(
            "### Create videos using Alpha AI"
        )

        video_prompt = gr.Textbox(

            label="Video Description",

            placeholder=(
                "A cinematic view of Saturn "
                "rotating in space..."
            )
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


# ==========================================
# 6. START SERVER
# ==========================================

app.queue(
    max_size=20
)

app.launch(

    server_name="0.0.0.0",

    server_port=int(
        os.environ.get("PORT", 10000)
    ),

    theme=gr.themes.Soft()
        )
