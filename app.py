import os
import base64
import io
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

CHAT_MODEL = "gemini-3.1-flash-lite"
IMAGE_MODEL = "gemini-nano-banana-2.1"


# -----------------------------
# ALPHA AI CHATBOT
# -----------------------------

def alpha_chat(message, history):

    try:
        if not message or not message.strip():
            yield "Please enter a message."
            return

        answer = ""

        stream = client.models.generate_content_stream(
            model=CHAT_MODEL,
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
        print("CHAT ERROR:", repr(e))
        yield f"Alpha AI Error: {str(e)}"

# -----------------------------
# ALPHA AI IMAGE GENERATOR
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
            return None, "No image was returned by Gemini."

        image_bytes = base64.b64decode(generated.data)

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image, "Image generated successfully!"

    except Exception as e:

        print("IMAGE ERROR:", repr(e))

        return None, (
            "Image generation failed. "
            "Please check the Render logs or try again."
        )


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

        gr.Markdown(
            "### Create images using artificial intelligence"
        )

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
