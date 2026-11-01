import os
import io
import time
import base64
import tempfile
import gradio as gr

from PIL import Image
from google import genai


# =====================================
# ALPHA AI CONFIGURATION
# =====================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in Render.")

client = genai.Client(api_key=API_KEY)


# MODELS

CHAT_MODEL = "gemini-3.5-flash-lite"

# UPDATED FAST IMAGE MODEL
IMAGE_MODEL = "gemini-3.1-flash-lite-image"

VIDEO_MODEL = "gemini-omni-1.1-flash"


# =====================================
# 1. ALPHA AI CHAT
# =====================================

def alpha_chat(message, history):

    if not message or not message.strip():
        return "Please enter a message."

    start = time.time()

    try:

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=message,
        )

        elapsed = time.time() - start

        print(f"CHAT TIME: {elapsed:.2f} seconds")

        return response.text or "No response generated."

    except Exception as e:

        print("CHAT ERROR:", repr(e))

        return f"Chat error: {str(e)}"


# =====================================
# 2. OPTIMIZED IMAGE GENERATOR
# =====================================

def generate_image(prompt):

    if not prompt or not prompt.strip():
        return None, "Enter an image description."

    start = time.time()

    try:

        print("ALPHA IMAGE GENERATION STARTED")

        # STEP 1: SEND IMAGE REQUEST

        api_start = time.time()

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

        api_time = time.time() - api_start

        print(
            f"IMAGE API TIME: {api_time:.2f} seconds"
        )

        # STEP 2: GET GENERATED IMAGE

        generated = result.output_image

        if not generated or not generated.data:

            return None, "No image was returned."

        # STEP 3: DECODE IMAGE

        decode_start = time.time()

        image_bytes = base64.b64decode(
            generated.data
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        image = image.convert("RGB")

        decode_time = time.time() - decode_start

        print(
            f"IMAGE PROCESSING TIME: {decode_time:.2f} seconds"
        )

        # STEP 4: TOTAL TIME

        total_time = time.time() - start

        print(
            f"TOTAL IMAGE TIME: {total_time:.2f} seconds"
        )

        return (
            image,
            f"Image generated successfully in {total_time:.2f} seconds!"
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

        upload_time = time.time() - start

        print(
            f"VIDEO UPLOAD TIME: {upload_time:.2f} seconds"
        )

        processing_start = time.time()

        while video_file.state.name == "PROCESSING":

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
            ]
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

    output_path = None

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
