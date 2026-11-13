import os
import io
import time
import tempfile
import mimetypes

import gradio as gr
import requests

from PIL import Image
from google import genai
from google.genai import types


# ==========================================
# CONFIGURATION
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-2.5-flash"
)

IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# GEMINI CHAT
# ==========================================

def ask_gemini(prompt, history=None):

    if not GEMINI_API_KEY:
        return "Gemini API key is missing."

    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )

        return response.text or "I couldn't generate a response."

    except Exception as e:
        return f"AI Error: {str(e)}"


# ==========================================
# VOICE INPUT
# ==========================================

def transcribe_audio(audio_path):

    if not audio_path:
        return ""

    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mimetypes.guess_type(audio_path)[0]
                    or "audio/wav"
                ),
                "Transcribe the spoken words accurately. "
                "Return only the transcription."
            ]
        )

        return response.text.strip() if response.text else ""

    except Exception:
        return ""


# ==========================================
# TEXT TO SPEECH
# ==========================================

def speak_text(text):

    if not text:
        return None

    try:
        from gtts import gTTS

        filename = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ).name

        speech = gTTS(
            text=text[:4000],
            lang="en"
        )

        speech.save(filename)

        return filename

    except Exception:
        return None


# ==========================================
# CHAT
# ==========================================

def chat(message, history):

    if not message:
        return history, None

    answer = ask_gemini(message)

    history = history or []

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    audio = speak_text(answer)

    return history, audio


def voice_chat(audio, history):

    text = transcribe_audio(audio)

    if not text:
        return history or [], None, "I couldn't understand the audio."

    updated_history, response_audio = chat(
        text,
        history
    )

    return updated_history, response_audio, text


# ==========================================
# IMAGE GENERATION
# ==========================================

def generate_image(prompt):

    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error("Cloudflare API credentials are missing.")

    if not prompt:
        raise gr.Error("Please enter an image prompt.")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"prompt": prompt},
            timeout=120
        )

        response.raise_for_status()

        image = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

        return image

    except Exception as e:
        raise gr.Error(f"Image generation failed: {e}")


def generate_image_voice(audio):

    prompt = transcribe_audio(audio)

    if not prompt:
        raise gr.Error("Could not understand your voice.")

    image = generate_image(prompt)

    return image, prompt


# ==========================================
# IMAGE ANALYSIS
# ==========================================

def analyze_image(image, question):

    if image is None:
        return "Please upload an image."

    try:
        buffer = io.BytesIO()

        image.save(buffer, format="PNG")

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=buffer.getvalue(),
                    mime_type="image/png"
                ),
                question or "Describe this image."
            ]
        )

        answer = response.text or "No answer generated."

        return answer

    except Exception as e:
        return f"Image analysis error: {e}"


# ==========================================
# VIDEO ANALYSIS ONLY
# ==========================================

def analyze_video(video_path, question):

    if not video_path:
        return "Please upload a video."

    try:
        uploaded_file = client.files.upload(
            file=video_path
        )

        for _ in range(30):

            file_info = client.files.get(
                name=uploaded_file.name
            )

            if file_info.state == "ACTIVE":
                break

            if file_info.state == "FAILED":
                return "Video processing failed."

            time.sleep(2)

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                file_info,
                question or "Analyze this video."
            ]
        )

        return response.text or "No analysis generated."

    except Exception as e:
        return f"Video analysis error: {e}"


# ==========================================
# VOICE ASSISTANT FOR OTHER FEATURES
# ==========================================

def voice_to_text(audio):

    text = transcribe_audio(audio)

    return text or "Voice transcription failed."


# ==========================================
# GRADIO INTERFACE
# ==========================================

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

footer {
    display: none !important;
}
"""


with gr.Blocks(
    title="Alpha AI",
    theme=gr.themes.Soft(),
    css=CSS
) as app:

    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits.
        Your AI assistant with voice capabilities.
        """
    )

    # --------------------------------------
    # CHAT
    # --------------------------------------

    with gr.Tab("AI Chat"):

        chatbot = gr.Chatbot(
            type="messages",
            height=450,
            label="Alpha Assistant"
        )

        with gr.Row():

            chat_input = gr.Textbox(
                placeholder="Ask Alpha anything...",
                scale=5
            )

            send_button = gr.Button(
                "Send",
                variant="primary"
            )

        chat_microphone = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak to Alpha"
        )

        recognized_text = gr.Textbox(
            label="Voice transcription"
        )

        response_audio = gr.Audio(
            label="Alpha Voice Response",
            autoplay=True
        )

        replay_button = gr.Button("Replay Response")

        clear_button = gr.Button("Clear Chat")

        send_button.click(
            chat,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, response_audio]
        )

        chat_input.submit(
            chat,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, response_audio]
        )

        chat_microphone.change(
            voice_chat,
            inputs=[chat_microphone, chatbot],
            outputs=[
                chatbot,
                response_audio,
                recognized_text
            ]
        )

        replay_button.click(
            speak_text,
            inputs=[recognized_text],
            outputs=[response_audio]
        )

        clear_button.click(
            lambda: ([], None),
            outputs=[chatbot, response_audio]
        )

    # --------------------------------------
    # IMAGE GENERATION
    # --------------------------------------

    with gr.Tab("Image Studio"):

        gr.Markdown("## Create Images with Alpha")

        image_prompt = gr.Textbox(
            label="Image Description",
            placeholder="Describe the image you want..."
        )

        image_voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak Your Image Prompt"
        )

        image_button = gr.Button(
            "Generate Image",
            variant="primary"
        )

        generated_image = gr.Image(
            label="Generated Image"
        )

        voice_prompt_button = gr.Button(
            "Generate from Voice"
        )

        voice_prompt_button.click(
            generate_image_voice,
            inputs=[image_voice],
            outputs=[generated_image, image_prompt]
        )

        image_button.click(
            generate_image,
            inputs=[image_prompt],
            outputs=[generated_image]
        )

    # --------------------------------------
    # IMAGE ANALYSIS
    # --------------------------------------

    with gr.Tab("Image Analysis"):

        uploaded_image = gr.Image(
            type="pil",
            label="Upload Image"
        )

        image_question = gr.Textbox(
            label="Your Question",
            placeholder="What is in this image?"
        )

        analysis_voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Ask by Voice"
        )

        transcribe_image_button = gr.Button(
            "Convert Voice to Text"
        )

        analyze_image_button = gr.Button(
            "Analyze Image",
            variant="primary"
        )

        image_result = gr.Textbox(
            label="Alpha's Analysis",
            lines=6
        )

        image_speech = gr.Audio(
            label="Listen to Analysis",
            autoplay=True
        )

        transcribe_image_button.click(
            voice_to_text,
            inputs=[analysis_voice],
            outputs=[image_question]
        )

        analyze_image_button.click(
            analyze_image,
            inputs=[uploaded_image, image_question],
            outputs=[image_result]
        ).then(
            speak_text,
            inputs=[image_result],
            outputs=[image_speech]
        )

    # --------------------------------------
    # VIDEO ANALYSIS
    # --------------------------------------

    with gr.Tab("Video Analysis"):

        gr.Markdown(
            "## Analyze Videos\n"
            "Upload a video and ask Alpha questions about it."
        )

        uploaded_video = gr.Video(
            label="Upload Video"
        )

        video_question = gr.Textbox(
            label="Your Question",
            placeholder="Explain this video..."
        )

        video_voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Ask Alpha by Voice"
        )

        video_transcribe_button = gr.Button(
            "Convert Voice to Text"
        )

        video_analyze_button = gr.Button(
            "Analyze Video",
            variant="primary"
        )

        video_result = gr.Textbox(
            label="Video Analysis",
            lines=7
        )

        video_speech = gr.Audio(
            label="Listen to Video Analysis",
            autoplay=True
        )

        video_transcribe_button.click(
            voice_to_text,
            inputs=[video_voice],
            outputs=[video_question]
        )

        video_analyze_button.click(
            analyze_video,
            inputs=[uploaded_video, video_question],
            outputs=[video_result]
        ).then(
            speak_text,
            inputs=[video_result],
            outputs=[video_speech]
        )


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port
        )
