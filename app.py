import os
import base64
import io
import time
import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# -----------------------------
# ALPHA AI CONFIGURATION
# -----------------------------

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in Render")

client = genai.Client(api_key=api_key)

CHAT_MODEL = "gemini-3.5-flash-lite"
IMAGE_MODEL = "gemini-nano-banana-2.1"


# -----------------------------
# CHATBOT WITH AUTOMATIC RETRY
# -----------------------------

def alpha_chat(message, history):

    if not message or not message.strip():
        return "Please enter a message."

    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=message,
                config=types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.7
                )
            )

            if response.text:
                return response.text

            return "Alpha couldn't generate a response."

        except Exception as e:

            print("CHAT ERROR:", repr(e))

            error_text = str(e).upper()
            error_code = getattr(e, "code", None)

            temporary_error = (
                error_code in (503, 429)
                or "503" in error_text
                or "429" in error_text
                or "UNAVAILABLE" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            if temporary_error and attempt < 2:
                wait_time = 2 ** (attempt + 1)
                print(f"Retrying Alpha AI in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            if temporary_error:
                return (
                    "Alpha AI is experiencing high demand. "
                    "Please try again shortly."
                )

            return "Alpha AI encountered an error. Please try again."

    return "Alpha AI is temporarily unavailable."


# -----------------------------
# IMAGE GENERATOR
# -----------------------------

def generate_image(prompt):

    try:
        if not prompt or not prompt.strip():
            return None, "Please enter an image description."

        interaction = client.interactions.create(
            model=IMAGE_MODEL,
            input=prompt.strip(),
            response_format={
                "type": "image",
                "aspect_ratio": "1:1",
                "image_size": "1K"
            }
        )

        generated = interaction.output_image

        if not generated or not generated.data:
            return None, "No image was returned."

        image_bytes = base64.b64decode(generated.data)

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image, "Image generated successfully!"

    except Exception as e:

        print("IMAGE ERROR:", repr(e))

        return None, "Image generation failed. Please try again."


# -----------------------------
# ALPHA AI INTERFACE
# -----------------------------

with gr.Blocks(title="ALPHA AI") as app:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits

        Your AI assistant for questions and image creation.
        """
    )

    with gr.Tab("Chat"):

        gr.ChatInterface(
            fn=alpha_chat,
            title="Alpha AI Chatbot",
            description="Ask Alpha AI anything."
        )

    with gr.Tab("Image Creator"):

        gr.Markdown("### Create images using artificial intelligence")

        prompt = gr.Textbox(
            label="Image Description",
            placeholder="A futuristic city on Mars..."
        )

        generate_button = gr.Button(
            "Generate Image",
            variant="primary"
        )

        output_image = gr.Image(
            label="Generated Image",
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

app.queue()

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
