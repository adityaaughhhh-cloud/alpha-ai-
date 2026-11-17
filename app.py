import os
import io
import time
import mimetypes
import tempfile
import logging
import ast
import operator
import re
from pathlib import Path

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types

# ============================================================
# ALPHA AI
# Chat, image generation, image analysis, video analysis,
# and push-to-talk voice assistance.
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha-ai")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash").strip()
CHAT_FALLBACK_MODEL = os.getenv("CHAT_FALLBACK_MODEL", "").strip()
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

MAX_CHAT_HISTORY = 12
VIDEO_PROCESSING_TIMEOUT = 120
REQUEST_TIMEOUT = 90
MAX_OUTPUT_TOKENS = 450

# A short system instruction keeps normal answers useful and reduces latency.
SYSTEM_INSTRUCTION = (
    "You are Alpha, a helpful AI assistant. Answer the user's exact question "
    "clearly and accurately. For ordinary questions, use about 2-5 concise "
    "sentences. For numerical questions, show the key calculation and final "
    "answer. Do not add unnecessary introductions or long explanations. "
    "Use plain text and Unicode symbols where possible. Do not wrap equations "
    "or ordinary text in LaTeX dollar signs."
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def require_gemini():
    if gemini_client is None:
        raise gr.Error("GEMINI_API_KEY is missing. Add it in Render > Environment, then redeploy Alpha AI.")
    return gemini_client


def clean_ai_formatting(text):
    """Convert common LaTeX output to readable plain text; remove raw delimiters."""
    text = str(text or "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$$", "")
    text = text.replace("$", "")

    replacements = {
        r"\\cdot": "·", r"\\times": "×", r"\\div": "÷",
        r"\\rightarrow": "→", r"\\to": "→", r"\\Rightarrow": "⇒",
        r"\\leq": "≤", r"\\geq": "≥", r"\\neq": "≠",
        r"\\approx": "≈", r"\\pm": "±", r"\\pi": "π",
        r"\\degree": "°", r"\\%": "%", r"\\infty": "∞",
        r"\\,": " ", r"\\;": " ", r"\\!": "",
    }
    for old, new in replacements.items():
        text = re.sub(old, new, text)

    text = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1/\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"√(\1)", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)

    # Convert common numeric subscripts/superscripts: CO2 -> CO₂, x^2 -> x².
    sub_digits = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    super_digits = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
    text = re.sub(r"([A-Za-z)])_\{?([0-9]+)\}?", lambda m: m.group(1) + m.group(2).translate(sub_digits), text)
    text = re.sub(r"([A-Za-z0-9)])\^\{?([0-9+\-=()]+)\}?", lambda m: m.group(1) + m.group(2).translate(super_digits), text)

    # Remove leftover braces used only for simple formatting.
    text = text.replace("{", "").replace("}", "")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_speech_file(text):
    text = (text or "").strip()
    if not text:
        return None
    output_path = None
    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", prefix="alpha_voice_", delete=False) as temp:
            output_path = temp.name
        gTTS(text=text[:2500], lang="en").save(output_path)
        return output_path
    except Exception:
        logger.exception("Text-to-speech failed")
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return None


def transcribe_audio(audio_path):
    if not audio_path:
        raise gr.Error("Please record a voice message first.")
    client = require_gemini()
    path = Path(str(audio_path))
    if not path.is_file():
        raise gr.Error("The microphone recording could not be found.")
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("audio/"):
        mime_type = {
            ".webm": "audio/webm", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
            ".ogg": "audio/ogg", ".wav": "audio/wav",
        }.get(path.suffix.lower(), "audio/wav")
    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
                "Transcribe the spoken words accurately. Return only the words spoken.",
            ],
            config=types.GenerateContentConfig(max_output_tokens=180, temperature=0),
        )
        return clean_ai_formatting(response.text or "")
    except Exception as exc:
        logger.exception("Audio transcription failed")
        raise gr.Error(f"Voice transcription failed: {exc}")


def _is_temporary_gemini_error(exc):
    message = str(exc).lower()
    return any(x in message for x in (
        "503", "429", "500", "502", "504", "unavailable", "high demand",
        "resource exhausted", "deadline exceeded", "temporarily", "connection reset",
        "timed out", "timeout",
    ))


def ask_gemini(contents):
    """Fast concise generation with one quick retry and optional fallback model."""
    client = require_gemini()
    models = [CHAT_MODEL]
    if CHAT_FALLBACK_MODEL and CHAT_FALLBACK_MODEL != CHAT_MODEL:
        models.append(CHAT_FALLBACK_MODEL)
    last_error = None
    for model_index, model in enumerate(models):
        attempts = 2 if model_index == 0 else 1
        for attempt in range(attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.3,
                    ),
                )
                return clean_ai_formatting(response.text or "") or "I couldn't generate a response."
            except Exception as exc:
                last_error = exc
                logger.warning("Gemini model %s attempt %s failed: %s", model, attempt + 1, exc)
                if not _is_temporary_gemini_error(exc):
                    raise gr.Error("Alpha could not process this request. Check the Gemini API key and model name.")
                if attempt == 0:
                    time.sleep(0.6)
    logger.error("All configured Gemini models failed: %s", last_error)
    raise gr.Error("Gemini is temporarily busy. Please try again shortly.")


# Fast local calculator for arithmetic expressions.
_MATH_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_math(node):
    if isinstance(node, ast.Expression):
        return _safe_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPERATORS:
        left, right = _safe_math(node.left), _safe_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent too large")
        result = _MATH_OPERATORS[type(node.op)](left, right)
        if abs(result) > 10**15:
            raise ValueError("Result too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPERATORS:
        return _MATH_OPERATORS[type(node.op)](_safe_math(node.operand))
    raise ValueError("Unsupported expression")


def quick_math_answer(message):
    text = (message or "").lower().replace(",", "")
    text = re.sub(r"\b(what is|what's|calculate|compute|solve|answer|please)\b", " ", text)
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    text = re.sub(r"\bplus\b", "+", text)
    text = re.sub(r"\bminus\b", "-", text)
    text = re.sub(r"\b(times|multiplied by)\b", "*", text)
    text = re.sub(r"\bdivided by\b", "/", text)
    text = text.strip().rstrip("?= ")
    if not re.fullmatch(r"[0-9.()+*/%\-\s]+", text) or not re.search(r"[+*/%\-]", text):
        return None
    try:
        value = _safe_math(ast.parse(text, mode="eval"))
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, SyntaxError):
        return None


def chat(message, history, speak=False):
    message = (message or "").strip()
    history = history or []
    if not message:
        return history, None, None

    quick_answer = quick_math_answer(message)
    if quick_answer is not None:
        answer = quick_answer
    else:
        conversation = []
        for item in history[-MAX_CHAT_HISTORY:]:
            if not isinstance(item, dict):
                continue
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                conversation.append(types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part.from_text(text=content)],
                ))
        conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        try:
            answer = ask_gemini(conversation)
        except gr.Error as exc:
            answer = str(exc)

    new_history = list(history)
    new_history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    # Text chat skips TTS so the user sees the answer immediately.
    audio_path = make_speech_file(answer) if speak else None
    return new_history, audio_path, audio_path


def voice_chat(audio_path, history):
    history = history or []
    try:
        transcript = transcribe_audio(audio_path)
        if not transcript:
            return history, None, "", None
        updated_history, response_audio, saved_audio = chat(transcript, history, speak=True)
        return updated_history, response_audio, transcript, saved_audio
    except gr.Error as exc:
        return history, None, str(exc), None


def replay_audio(audio_path):
    return audio_path if audio_path and os.path.isfile(audio_path) else None


def stop_audio():
    return None


def clear_chat():
    return [], None, None, "", None


# Image generation (Cloudflare FLUX)
def generate_image(prompt):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter an image description.")
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error("Add CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in Render Environment.")
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}", "Content-Type": "application/json"}
    try:
        response = requests.post(endpoint, headers=headers, json={"prompt": prompt}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        if "application/json" in response.headers.get("content-type", "").lower():
            try:
                detail = response.json().get("errors") or response.json().get("messages") or response.json()
            except ValueError:
                detail = response.text[:500]
            raise gr.Error(f"Cloudflare did not return an image: {detail}")
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
    return generate_image(prompt), prompt


# Image analysis
def analyze_image(image, question):
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        image_part = types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")
        prompt = (question or "").strip() or "Describe this image."
        return ask_gemini([image_part, prompt])
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise gr.Error(f"Image analysis failed: {exc}")


# Video analysis only
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
            raise gr.Error("Video processing timed out. Please try a shorter video.")
        prompt = (question or "").strip() or "Summarize and analyze this video."
        return ask_gemini([file_info, prompt])
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Video analysis failed")
        raise gr.Error(f"Video analysis failed: {exc}")
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                logger.warning("Could not delete temporary Gemini video file.")


def voice_to_text(audio_path):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        raise gr.Error("Alpha could not understand the recording.")
    return transcript


CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
footer { display: none !important; }
#alpha-mic { min-width: 76px !important; max-width: 92px !important; align-self: end !important; }
#alpha-mic label { display: none !important; }
#alpha-mic .wrap { min-height: 54px !important; border-radius: 18px !important; border: 1px solid #2563eb !important; background: #2563eb !important; overflow: hidden !important; }
#alpha-mic button { color: white !important; }
#alpha-mic audio { max-width: 100% !important; }
#alpha-send button { min-height: 54px !important; border-radius: 16px !important; }
#alpha-chat-input textarea { min-height: 54px !important; border-radius: 16px !important; }
.alpha-note { opacity: 0.75; font-size: 0.9rem; }
"""

with gr.Blocks(title="Alpha AI") as app:
    gr.Markdown("# ALPHA AI\n### Think Beyond Limits.\nChat, create images, analyze images and videos, and talk to Alpha.")

    with gr.Tab("AI Chat"):
        chatbot = gr.Chatbot(height=450, label="Alpha Assistant")
        with gr.Row(equal_height=True):
            chat_input = gr.Textbox(placeholder="Message Alpha...", label="", show_label=False, scale=6, elem_id="alpha-chat-input", lines=1)
            chat_microphone = gr.Audio(sources=["microphone"], type="filepath", label=None, show_label=False, container=False, elem_id="alpha-mic", scale=1)
            send_button = gr.Button("Send", variant="primary", scale=1, elem_id="alpha-send")
        gr.Markdown("Tap the microphone control to record a message. When recording is submitted, Alpha transcribes it and replies.", elem_classes=["alpha-note"])
        recognized_text = gr.Textbox(label="Voice transcription", interactive=False, visible=False)
        response_audio = gr.Audio(label="Alpha voice response", autoplay=True)
        last_chat_audio = gr.State(None)
        with gr.Row():
            replay_button = gr.Button("Replay response")
            stop_button = gr.Button("⏹ Stop audio")
            clear_button = gr.Button("Clear chat")

        send_event = send_button.click(fn=chat, inputs=[chat_input, chatbot], outputs=[chatbot, response_audio, last_chat_audio])
        send_event.then(fn=lambda: "", inputs=None, outputs=[chat_input])
        submit_event = chat_input.submit(fn=chat, inputs=[chat_input, chatbot], outputs=[chatbot, response_audio, last_chat_audio])
        submit_event.then(fn=lambda: "", inputs=None, outputs=[chat_input])
        chat_microphone.change(fn=voice_chat, inputs=[chat_microphone, chatbot], outputs=[chatbot, response_audio, recognized_text, last_chat_audio])
        replay_button.click(fn=replay_audio, inputs=[last_chat_audio], outputs=[response_audio])
        stop_button.click(fn=stop_audio, inputs=None, outputs=[response_audio])
        clear_button.click(fn=clear_chat, inputs=None, outputs=[chatbot, response_audio, last_chat_audio, recognized_text, chat_input])

    with gr.Tab("Image Studio"):
        gr.Markdown("## Create images with Alpha")
        image_prompt = gr.Textbox(label="Image description", placeholder="Describe the image you want...")
        image_voice = gr.Audio(sources=["microphone"], type="filepath", label="Speak your image prompt")
        with gr.Row():
            image_button = gr.Button("Generate image", variant="primary")
            voice_prompt_button = gr.Button("Generate from voice")
        generated_image = gr.Image(label="Generated image", type="pil")
        image_button.click(fn=generate_image, inputs=[image_prompt], outputs=[generated_image])
        voice_prompt_button.click(fn=generate_image_from_voice, inputs=[image_voice], outputs=[generated_image, image_prompt])

    with gr.Tab("Image Analysis"):
        uploaded_image = gr.Image(type="pil", label="Upload image")
        image_question = gr.Textbox(label="Your question", placeholder="What is in this image?")
        analysis_voice = gr.Audio(sources=["microphone"], type="filepath", label="Ask Alpha by voice")
        transcribe_image_button = gr.Button("Convert voice to text")
        analyze_image_button = gr.Button("Analyze image", variant="primary")
        image_result = gr.Textbox(label="Alpha's analysis", lines=6)
        image_speech = gr.Audio(label="Listen to analysis", autoplay=True)
        transcribe_image_button.click(fn=voice_to_text, inputs=[analysis_voice], outputs=[image_question])
        analyze_image_button.click(fn=analyze_image, inputs=[uploaded_image, image_question], outputs=[image_result]).then(fn=make_speech_file, inputs=[image_result], outputs=[image_speech])

    with gr.Tab("Video Analysis"):
        gr.Markdown("## Analyze videos\nUpload a video and ask Alpha questions about its contents.")
        uploaded_video = gr.Video(label="Upload video")
        video_question = gr.Textbox(label="Your question", placeholder="Summarize this video...")
        video_voice = gr.Audio(sources=["microphone"], type="filepath", label="Ask Alpha by voice")
        video_transcribe_button = gr.Button("Convert voice to text")
        video_analyze_button = gr.Button("Analyze video", variant="primary")
        video_result = gr.Textbox(label="Video analysis", lines=7)
        video_speech = gr.Audio(label="Listen to video analysis", autoplay=True)
        video_transcribe_button.click(fn=voice_to_text, inputs=[video_voice], outputs=[video_question])
        video_analyze_button.click(fn=analyze_video, inputs=[uploaded_video, video_question], outputs=[video_result]).then(fn=make_speech_file, inputs=[video_result], outputs=[video_speech])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(), css=CSS)
    
