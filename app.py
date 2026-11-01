import os
import io
import time
import base64
import tempfile
import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# =====================================
# ALPHA AI 2.0 CONFIGURATION
# =====================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in Render.")

client = genai.Client(api_key=API_KEY)


# =====================================
# MODEL CONFIGURATION
# =====================================

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gemini-3.8-flash"
)

BACKUP_MODEL = os.environ.get(
    "BACKUP_MODEL",
    "gemini-flash-latest"
)

IMAGE_MODEL = "gemini-3.1-flash-lite-image"

VIDEO_MODEL = "gemini-omni-1.1-flash"


# =====================================
# CHAT PERFORMANCE SETTINGS
# =====================================

MAX_OUTPUT_TOKENS = 512

SYSTEM_INSTRUCTION = """
You are Alpha AI, a helpful and intelligent AI assistant.

Instructions:
1. Answer questions accurately.
2. Keep simple answers short.
3. Explain difficult questions step by step.
4. Use mathematical reasoning when necessary.
5. Avoid unnecessary introductions.
6. Use clear and simple language.
7. Never invent facts.
"""


# =====================================
# ERROR DETECTION
# =====================================

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
        "internal server error"
    ]

    return any(
        item in message
        for item in temporary_errors
    )


# =====================================
# CHAT HISTORY CONVERSION
# =====================================

def prepare_chat_contents(message, history):

    contents = []

    if history:

        for item in history[-10:]:

            if isinstance(item, dict):

                role = item.get("role")
                content = item.get("content")

                if isinstance(content, str) and role in [
                    "user",
                    "assistant"
                ]:

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
                            role="assistant",
                            parts=[
                                types.Part.from_text(
                                    text=assistant_message
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


# =====================================
# 1. ALPHA AI FAST CHAT
# =====================================

def alpha_chat(message, history):

    if not message or not message.strip():

        yield "Please enter a message."
        return

    start = time.time()

    contents = prepare_chat_contents(
        message.strip(),
        history
    )

    models = [
        CHAT_MODEL,
        BACKUP_MODEL
    ]

    for model_index, model in enumerate(models):

        response_text = ""

        try:

            print(
                f"ALPHA CHAT STARTED: {model}"
            )

            stream = client.models.generate_content_stream(

                model=model,

                contents=contents,

                config=types.GenerateContentConfig(

                    system_instruction=SYSTEM_INSTRUCTION,

                    temperature=0.4,

                    max_output_tokens=MAX_OUTPUT_TOKENS,

                    candidate_count=1,

                    thinking_config=types.ThinkingConfig(
                        thinking_level="LOW"
                    )
                )
            )

            for chunk in stream:

                text = chunk.text

                if text:

                    response_text += text

                    # Send partial response immediately.
                    yield response_text

            elapsed = time.time() - start

            print(
                f"CHAT MODEL: {model}"
            )

            print(
                f"TOTAL CHAT TIME: {elapsed:.2f} seconds"
            )

            if not response_text:

                yield "Alpha AI could not generate a response."

            return

        except Exception as e:

            print(
                f"CHAT ERROR ({model}):",
                repr(e)
            )

            # Do not restart a response after
            # some text has already been displayed.

            if response_text:

                yield (
                    response_text
                    + "\n\n[Connection interrupted. Please retry.]"
                )

                return

            # Try backup only for temporary errors.

            if (
                is_temporary_error(e)
                and model_index == 0
            ):

                yield "Alpha AI is switching to its backup model..."

                time.sleep(1)

                continue

            yield (
                "Alpha AI temporarily cannot answer. "
                "Please try again shortly.\n\n"
                + str(e)
            )

            return


# =====================================
# 2. IMAGE GENERATOR
# =====================================

def generate_image(prompt):

    if not prompt or not prompt.strip():

        return None, "Enter an image description."

    start = time.time()

    try:

        print("ALPHA IMAGE GENERATION STARTED")

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

        elapsed = time.time() - start

        print(
            f"IMAGE TIME: {elapsed:.2f} seconds"
        )

        return (
            image,
            f"Image generated in {elapsed:.2f} seconds!"
        )

    except Exception as e:

        print("IMAGE ERROR:", repr(e))

        return None, f"Image error: {str(e)}"


# =====================================
# 3. VIDEO ANALYZER
# =====================================

def analyze_video(video, question):

    try:

        if not video:

            return "Please upload a video."

        if not question or not question.strip():

            question = "Explain what happens in this video."

        start = time.time()

        video_file = client.files.upload(
            file=video
        )

        processing_start = time.time()

        while (
            not video_file.state
            or video_file.state.name == "PROCESSING"
        ):

            if time.time() - processing_start > 300:

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

        elapsed = time.time() - start

        print(
            f"VIDEO ANALYSIS TIME: {elapsed:.2f} seconds"
        )

        return response.text or "No analysis generated."

    except Exception as e:

        print("VIDEO ANALYSIS ERROR:", repr(e))

        return f"Video analysis error: {str(e)}"


# =====================================
# 4. VIDEO GENERATOR
# =====================================

def generate_video(prompt):

    start = time.time()

    try:

        if not prompt or not prompt.strip():

            return None, "Enter a video description."

        print("ALPHA VIDEO GENERATION STARTED")

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

            output_file.write(video_bytes)

            output_path = output_file.name

        elapsed = time.time() - start

        print(
            f"VIDEO GENERATION TIME: {elapsed:.2f} seconds"
        )

        return (
            output_path,
            f"Video generated in {elapsed:.2f} seconds!"
        )

    except Exception as e:

        print("VIDEO GENERATION ERROR:", repr(e))

        return None, f"Video generation error: {str(e)}"


# =====================================
# 5. ALPHA AI INTERFACE
# =====================================

with gr.Blocks(title="ALPHA AI") as app:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits

        Your AI assistant for chat, image creation,
        video analysis and video creation.
        """
    )

    # CHAT

    with gr.Tab("Chat"):

        gr.ChatInterface(

            fn=alpha_chat,

            title="Alpha AI Chatbot",

            description="Ask Alpha AI anything."
        )

    # IMAGE CREATOR

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

    # VIDEO ANALYZER

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

    # VIDEO GENERATOR

    with gr.Tab("Video Generator"):

        gr.Markdown(
            "### Create videos using Alpha AI"
        )

        video_prompt = gr.Textbox(
            label="Video Description",
            placeholder="A cinematic view of Saturn rotating in space..."
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


# =====================================
# 6. START SERVER
# =====================================

app.queue()

app.launch(

    server_name="0.0.0.0",

    server_port=int(
        os.environ.get("PORT", 10000)
    )
)
