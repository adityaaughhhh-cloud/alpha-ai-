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
# ALPHA AI CONFIGURATION
# =====================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in Render.")

client = genai.Client(api_key=API_KEY)

CHAT_MODEL = "gemini-3.5-flash-lite"
IMAGE_MODEL = "gemini-nano-banana-2.1"
VIDEO_MODEL = "gemini-omni-1.1-flash"


# =====================================
# 1. ALPHA AI CHAT
# =====================================

def alpha_chat(message, history):

    if not message or not message.strip():
        return "Please enter a message."

    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.7
            )
        )

        return response.text or "No response generated."

    except Exception as e:
        print("CHAT ERROR:", repr(e))
        return f"Chat error: {str(e)}"


# =====================================
# 2. IMAGE GENERATOR
# =====================================

def generate_image(prompt):

    try:
        if not prompt or not prompt.strip():
            return None, "Enter an image description."

        result = client.interactions.create(
            model=IMAGE_MODEL,
            input=prompt.strip(),
            response_format={
                "type": "image",
                "aspect_ratio": "1:1",
                "image_size": "1K"
            }
        )

        generated = result.output_image

        if not generated or not generated.data:
            return None, "No image was returned."

        image_bytes = base64.b64decode(generated.data)

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image, "Image generated successfully!"

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

        video_file = client.files.upload(file=video)

        start_time = time.time()

        while video_file.state.name == "PROCESSING":

            if time.time() - start_time > 300:
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

        return response.text or "No analysis generated."

    except Exception as e:
        print("VIDEO ANALYSIS ERROR:", repr(e))
        return f"Video analysis error: {str(e)}"


# =====================================
# 4. VIDEO GENERATOR
# =====================================

def generate_video(prompt):

    output_path = None
    start_time = time.time()

    try:
        if not prompt or not prompt.strip():
            return None, "Enter a video description."

        print("Starting Alpha AI video generation...")

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
            return None, "No video data returned by Gemini."

        # Decode generated video

        video_bytes = base64.b64decode(
            video_output.data
        )

        # Save video as MP4

        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as output_file:

            output_file.write(video_bytes)
            output_path = output_file.name

        elapsed = time.time() - start_time

        print(
            f"Video generated in {elapsed:.2f} seconds."
        )

        if elapsed <= 60:
            status = (
                f"Video generated in {elapsed:.2f} seconds!"
            )
        else:
            status = (
                f"Video generated in {elapsed:.2f} seconds. "
                "The 60-second target was exceeded."
            )

        return output_path, status

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

        Your AI assistant for chat, image creation
        and video creation.
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

        gr.Markdown("### Create images using AI")

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
            "### Create videos using AI"
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
