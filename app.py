import os
import base64
import io
import gradio as gr

from PIL import Image
from google import genai
from google.genai import types

# Get API key from Render Environment Variables
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in Render")

# Initialize Gemini
client = genai.Client(api_key=api_key)

# Chat model
model_name = "gemini-3.5-flash-lite"


# -----------------------------
# ALPHA AI CHATBOT
# -----------------------------

def alpha_chat(message, history):
    try:
        answer = ""

        stream = client.models.generate_content_stream(
            model=model_name,
            contents=message,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.7
            )
        )

        for chunk in stream:
            if chunk.text:
                answer += chunk.text
                yield answer

        if not answer:
            yield "Alpha couldn't generate a response."

    except Exception as e:
        yield f"ALPHA ERROR: {str(e)}"


# -----------------------------
# ALPHA AI IMAGE CREATOR
# -----------------------------

def generate_image(prompt):
    import time

    start = time.time()

    try:
        if not prompt or not prompt.strip():
            return None, "Enter an image description."

        interaction = client.interactions.create(
            model="gemini-3.1-flash-lite-image",
            input=prompt,
            generation_config={
                "thinking_level": "minimal"
            },
            response_format={
                "type": "image",
                "aspect_ratio": "1:1",
                "image_size": "1K"
            }
        )

        if not interaction.output_image:
            return None, "No image returned."

        image_bytes = base64.b64decode(
            interaction.output_image.data
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        elapsed = time.time() - start

        return image, f"Generated in {elapsed:.2f} seconds!"

    except Exception as e:
        return None, f"IMAGE ERROR: {str(e)}"


# -----------------------------
# ALPHA AI INTERFACE
# -----------------------------

with gr.Blocks(title="ALPHA AI") as app:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits
        """
    )

    with gr.Tab("Chat"):
        gr.ChatInterface(
            fn=alpha_chat,
            title="Alpha AI Chatbot",
            description="Ask Alpha AI anything."
        )

    with gr.Tab("Image Creator"):
        gr.Markdown("### Create images using AI")

        prompt = gr.Textbox(
            label="Describe your image",
            placeholder="A futuristic city on Mars..."
        )

        generate_button = gr.Button("Generate Image")

        output_image = gr.Image(
            label="Your Generated Image",
            type="pil"
        )

        status = gr.Markdown()

        generate_button.click(
            fn=generate_image,
            inputs=prompt,
            outputs=[output_image, status]
        )


# -----------------------------
# START SERVER
# -----------------------------

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
        )
