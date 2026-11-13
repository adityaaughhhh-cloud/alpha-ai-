import os
import io
import time
import mimetypes
import tempfile
import logging
from pathlib import Path

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI — Chat, Image Generation, Image Analysis,
# Video Analysis, and Voice Assistance.
#
# Video GENERATION is intentionally not included.
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha-ai")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash").strip()
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

MAX_CHAT_HISTORY = 20
VIDEO_PROCESSING_TIMEOUT = 120
REQUEST_TIMEOUT = 120

# Create the Gemini client only when a key is available.
# This prevents the app from crashing during startup if a key is missing.
gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# ------------------------------
# Shared helpers
# ------------------------------

def require_gemini():
    if gemini_client is None:
        raise gr.Error(
            "GEMINI_API_KEY is missing. Add it in Render → Environment, "
            "then redeploy Alpha AI."
        )
    return gemini_client


def make_speech_file(text):
    """Convert text to an MP3 file and return its path."""
    text = (text or "").strip()
    if not text:
        return None

    try:
        from gtts import gTTS

        temp = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            prefix="alpha_voice_",
            delete=False
        )
        output_path = temp.name
        temp.close()

        gTTS(text=text[:4000], lang="en").save(output_path)
        return output_path

    except Exception:
        logger.exception("Text-to-speech failed")
        return None


def transcribe_audio(audio_path):
    """Transcribe a Gradio microphone recording with Gemini."""
    if not audio_path:
        return ""

    client = require_gemini()
    path = Path(str(audio_path))

    if not path.is_file():
        raise gr.Error("The microphone recording could not be found.")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("audio/"):
        mime_type = "audio/wav"

    try:
        audio_bytes = path.read_bytes()
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ),
                (
                    "Transcribe the spoken words accurately. "
                    "Return only the words that were spoken. "
                    "Do not answer the speaker's question."
                )
            ]
        )
        return (response.text or "").strip()

    except Exception as exc:
        logger.exception("Audio transcription failed")
        raise gr.Error(f"Voice transcription failed: {exc}")


def ask_gemini(contents):
    """Send text, image, or video contents to Gemini."""
    client = require_gemini()

    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=contents
        )
        return (response.text or "").strip() or "I couldn't generate a response."

    except Exception as exc:
        logger.exception("Gemini request failed")
        raise gr.Error(f"Alpha AI request failed: {exc}")


# ------------------------------
# Chat
# ------------------------------

def chat(message, history):
    message = (message or "").strip()
    history = history or []

    if not message:
        return history, None, None

    # Include recent conversation context so follow-up questions make sense.
    conversation = []
    for item in history[-MAX_CHAT_HISTORY:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        conversation.append(
            types.Content(
                role="user" if role == "user" else "model",
                parts=[types.Part.from_text(text=content)]
            )
        )

    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)]
        )
    )

    try:
        answer = ask_gemini(conversation)
    except gr.Error as exc:
        answer = str(exc)

    new_history = list(history)
    new_history.append({"role": "user", "content": message})
    new_history.append({"role": "assistant", "content": answer})

    audio_path = make_speech_file(answer)
    return new_history, audio_path, audio_path


def voice_chat(audio_path, history):
    history = history or []

    try:
        transcript = transcribe_audio(audio_path)
        if not transcript:
            return history, None, "", None

        updated_history, response_audio, saved_audio = chat(
            transcript,
            history
        )
        return updated_history, response_audio, transcript, saved_audio

    except gr.Error as exc:
        return history, None, str(exc), None


def replay_audio(audio_path):
    return audio_path if audio_path and os.path.isfile(audio_path) else None


def clear_chat():
    return [], None, None, ""


# ------------------------------
# Image generation (Cloudflare FLUX)
# ------------------------------

def generate_image(prompt):
    prompt = (prompt or "").strip()

    if not prompt:
        raise gr.Error("Please enter an image description.")

    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error(
            "Cloudflare credentials are missing. Add "
            "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in Render."
        )

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json={"prompt": prompt},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        # FLUX normally returns image bytes. Handle JSON error responses clearly.
        if "application/json" in content_type:
            try:
                data = response.json()
                message = data.get("errors") or data.get("messages") or data
            except ValueError:
                message = response.text[:500]
            raise gr.Error(f"Cloudflare did not return an image: {message}")

        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    except gr.Error:
        raise
    except requests.RequestException as exc:
        logger.exception("Cloudflare image request failed")
        raise gr.Error(f"Image generation request failed: {exc}")
    except Exception as exc:
        logger.exception("Image decoding failed")
        raise gr.Error(f"Could not read the generated image: {exc}")


def generate_image_from_voice(audio_path):
    prompt = transcribe_audio(audio_path)
    if not prompt:
        raise gr.Error("Alpha could not understand the voice prompt.")

    image = generate_image(prompt)
    return image, prompt


# ------------------------------
# Image analysis
# ------------------------------

def analyze_image(image, question):
    if image is None:
        raise gr.Error("Please upload an image first.")

    buffer = io.BytesIO()
    try:
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        image_part = types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/jpeg"
        )

        prompt = (question or "").strip() or "Describe this image."
        return ask_gemini([image_part, prompt])

    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise gr.Error(f"Image analysis failed: {exc}")


# ------------------------------
# Video analysis ONLY
# ------------------------------

def analyze_video(video_path, question):
    if not video_path:
        raise gr.Error("Please upload a video first.")

    client = require_gemini()
    uploaded_file = None

    try:
        uploaded_file = client.files.upload(file=video_path)
        deadline = time.time() + VIDEO_PROCESSING_TIMEOUT
        file_info = uploaded_file

        while time.time() < deadline:
            file_info = client.files.get(name=uploaded_file.name)
            state = getattr(file_info, "state", "")
            state_name = getattr(state, "name", str(state)).upper()

            if "ACTIVE" in state_name:
                break

            if "FAILED" in state_name:
                raise gr.Error("Gemini could not process this video.")

            time.sleep(2)
        else:
            raise gr.Error(
                "Video processing timed out. Please try a shorter video."
            )

        prompt = (question or "").strip() or "Summarize and analyze this video."
        return ask_gemini([file_info, prompt])

    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Video analysis failed")
        raise gr.Error(f"Video analysis failed: {exc}")

    finally:
        # Clean up the temporary Gemini upload when possible.
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                logger.warning("Could not delete temporary Gemini video file.")


# ------------------------------
# Voice transcription for feature tabs
# ------------------------------

def voice_to_text(audio_path):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        raise gr.Error("Alpha could not understand the recording.")
    return transcript


# ------------------------------
# Gradio interface
# ------------------------------

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(title="Alpha AI") as app:
    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits.
        Chat, create images, analyze images and videos, and talk to Alpha.
        """
    )

    # CHAT
    with gr.Tab("AI Chat"):
        chatbot = gr.Chatbot(
            height=450,
            label="Alpha Assistant"
        )

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask Alpha anything...",
                label="Message",
                scale=5
            )
            send_button = gr.Button("Send", variant="primary")

        chat_microphone = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak to Alpha"
        )

        recognized_text = gr.Textbox(
            label="Voice transcription",
            interactive=False
        )

        response_audio = gr.Audio(
            label="Alpha voice response",
            autoplay=True
        )

        last_chat_audio = gr.State(None)

        with gr.Row():
            replay_button = gr.Button("Replay response")
            clear_button = gr.Button("Clear chat")

        send_button.click(
            fn=chat,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, response_audio, last_chat_audio]
        )
        chat_input.submit(
            fn=chat,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, response_audio, last_chat_audio]
        )
        chat_microphone.change(
            fn=voice_chat,
            inputs=[chat_microphone, chatbot],
            outputs=[
                chatbot,
                response_audio,
                recognized_text,
                last_chat_audio
            ]
        )
        replay_button.click(
            fn=replay_audio,
            inputs=[last_chat_audio],
            outputs=[response_audio]
        )
        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[
                chatbot,
                response_audio,
                last_chat_audio,
                recognized_text
            ]
        )

    # IMAGE GENERATION
    with gr.Tab("Image Studio"):
        gr.Markdown("## Create images with Alpha")

        image_prompt = gr.Textbox(
            label="Image description",
            placeholder="Describe the image you want..."
        )

        image_voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your image prompt"
        )

        with gr.Row():
            image_button = gr.Button("Generate image", variant="primary")
            voice_prompt_button = gr.Button("Generate from voice")

        generated_image = gr.Image(
            label="Generated image",
            type="pil"
        )

        image_button.click(
            fn=generate_image,
            inputs=[image_prompt],
            outputs=[generated_image]
        )
        voice_prompt_button.click(
            fn=generate_image_from_voice,
            inputs=[image_voice],
            outputs=[generated_image, image_prompt]
        )

    # IMAGE ANALYSIS
    with gr.Tab("Image Analysis"):
        uploaded_image = gr.Image(
            type="pil",
            label="Upload image"
        )

        image_question = gr.Textbox(
            label="Your question",
            placeholder="What is in this image?"
        )

        analysis_voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Ask Alpha by voice"
        )

        transcribe_image_button = gr.Button("Convert voice to text")
        analyze_image_button = gr.Button("Analyze image", variant="primary")

        image_result = gr.Textbox(
            label="Alpha's analysis",
            lines=6
        )

        image_speech = gr.Audio(
            label="Listen to analysis",
            autoplay=True
        )

        transcribe_image_button.click(
            fn=voice_to_text,
            inputs=[analysis_voice],
            outputs=[image_question]
        )

        analyze_image_button.click(
            fn=analyze_image,
            inputs=[uploaded_image, image_question],
            outputs=[image_result]
        ).then(
            fn=make_speech_file,
            inputs=[image_result],
            outputs=[image_speech]
        )

    # VIDEO ANALYSIS — NO VIDEO GENERATION
    with gr.Tab("Video Analysis"):
        gr.Markdown(
            "## Analyze videos\n"
            "Upload a video and ask Alpha questions about its contents."
        )

        uploaded_video = gr.Video(label="Upload video")

        video_question = gr.Textbox(
            label="Your question",
            placeholder="Summarize this video..."
        )

        video_voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Ask Alpha by voice"
        )

        video_transcribe_button = gr.Button("Convert voice to text")
        video_analyze_button = gr.Button("Analyze video", variant="primary")

        video_result = gr.Textbox(
            label="Video analysis",
            lines=7
        )

        video_speech = gr.Audio(
            label="Listen to video analysis",
            autoplay=True
        )

        video_transcribe_button.click(
            fn=voice_to_text,
            inputs=[video_voice],
            outputs=[video_question]
        )

        video_analyze_button.click(
            fn=analyze_video,
            inputs=[uploaded_video, video_question],
            outputs=[video_result]
        ).then(
            fn=make_speech_file,
            inputs=[video_result],
            outputs=[video_speech]
        )


# ------------------------------
# Start server (Render compatible)
# ------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        css=CSS
    )
    
