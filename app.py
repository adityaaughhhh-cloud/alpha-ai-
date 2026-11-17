import os
import io
import time
import base64
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
# Chat, image generation, image analysis, and video analysis.
# Voice features have been removed.
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha-ai")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

# Primary + automatic fallback chain. All are official stable Gemini API IDs.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite").strip()
_fallback_env = os.getenv(
    "CHAT_FALLBACK_MODELS",
    "gemini-3.8-flash,gemini-3.1-flash-lite"
)
CHAT_FALLBACK_MODELS = [
    model.strip() for model in _fallback_env.split(",") if model.strip()
]
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

MAX_CHAT_HISTORY = 12
VIDEO_PROCESSING_TIMEOUT = 120
REQUEST_TIMEOUT = 120
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "384"))
IMAGE_STEPS = 4  # FLUX.1 Schnell default; lower latency than higher step counts

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ------------------------------
# Shared helpers
# ------------------------------

def require_gemini():
    if gemini_client is None:
        raise gr.Error(
            "GEMINI_API_KEY is missing. Add it in Render > Environment, "
            "then redeploy Alpha AI."
        )
    return gemini_client


def clean_ai_text(text):
    """Convert common raw LaTeX markup into readable plain Unicode text."""
    if not text:
        return ""

    text = str(text)
    replacements = {
        r"\\rightarrow": " → ",
        r"\\to": " → ",
        r"\\times": " × ",
        r"\\div": " ÷ ",
        r"\\pm": " ± ",
        r"\\neq": " ≠ ",
        r"\\leq": " ≤ ",
        r"\\geq": " ≥ ",
        r"\\approx": " ≈ ",
        r"\\Delta": "Δ",
        r"\\alpha": "α",
        r"\\beta": "β",
        r"\\pi": "π",
        r"\\infty": "∞",
        r"\\cdot": " · ",
        r"\\degree": "°",
        r"\\%": "%",
        r"\\ ": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove common LaTeX wrappers and formatting commands.
    text = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\(?:displaystyle|textstyle)\b", "", text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace("$$", "").replace("$", "")

    # Convert simple chemical/math subscripts, e.g. CO_{2}, H_2O.
    sub_map = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    text = re.sub(
        r"_\{([0-9+\-=()]+)\}|_([0-9])",
        lambda m: (m.group(1) or m.group(2)).translate(sub_map),
        text,
    )

    # Convert common powers, e.g. x^{2}, m^2.
    sup_map = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
    text = re.sub(
        r"\^\{([0-9+\-=()]+)\}|\^([0-9])",
        lambda m: (m.group(1) or m.group(2)).translate(sup_map),
        text,
    )

    # Remove leftover LaTeX command slashes while retaining their words.
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _is_retryable_gemini_error(exc):
    """Identify temporary overload, rate-limit, network and model-routing errors."""
    message = str(exc).lower()
    markers = (
        "503", "429", "500", "502", "504", "unavailable",
        "high demand", "resource exhausted", "deadline exceeded",
        "temporarily", "connection reset", "timed out", "timeout",
        "model not found", "not found", "unknown model",
        "is not supported", "unsupported model",
    )
    return any(marker in message for marker in markers)


def ask_gemini(contents):
    """Try the primary model and stable fallbacks instead of stopping on one 503."""
    client = require_gemini()
    models = []
    for model in [CHAT_MODEL, *CHAT_FALLBACK_MODELS]:
        if model and model not in models:
            models.append(model)

    last_error = None
    for model_index, model in enumerate(models):
        # Try each configured model once to avoid wasting time on repeated retries.
        for attempt in range(1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.4,
                        system_instruction=(
                            "You are Alpha, a fast, helpful and accurate assistant. "
                            "Answer the user's actual request directly. If they ask "
                            "for a study routine, give a usable routine and explain "
                            "the key choices briefly, not just a bare timetable. "
                            "Use relevant Unicode emojis naturally in normal replies. "
                            "If the user asks for emojis or emoji art, output the requested "
                            "emojis clearly, not a description of them. Do not replace "
                            "emojis with words. Be concise unless more detail is requested. "
                            "Use readable Unicode symbols such as CO₂ and H₂O; do not use raw "
                            "LaTeX dollar delimiters. Show calculation steps when useful."
                        ),
                    ),
                )
                answer = clean_ai_text(getattr(response, "text", "") or "")
                if answer:
                    logger.info("Chat answered using model %s", model)
                    return answer
                raise RuntimeError(f"Gemini model {model} returned an empty response")
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini model %s attempt %s/%s failed: %s",
                    model, attempt + 1, 1, exc
                )
                if not _is_retryable_gemini_error(exc):
                    # Invalid model IDs and temporary provider failures can be routed
                    # to another model; authentication/configuration errors cannot.
                    msg = str(exc).lower()
                    if not any(x in msg for x in (
                        "model not found", "not found", "unknown model",
                        "unsupported model", "is not supported",
                    )):
                        raise gr.Error(
                            "Alpha could not reach Gemini. Check that GEMINI_API_KEY "
                            "is valid and that the Gemini API is enabled for this key."
                        )
                    break
                break

    logger.error("Every configured Gemini chat model failed: %s", last_error)
    raise gr.Error(
        "Alpha's AI providers are temporarily unavailable or busy. "
        "Please retry in a moment. If this keeps happening, check the Gemini "
        "API quota, billing status, and Render logs."
    )


# ------------------------------
# Chat and fast local calculator
# ------------------------------

_MATH_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
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
    text = message.lower().replace(",", "")
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


EMOJI_SETS = {
    "faces": "😀 😃 😄 😁 😆 😅 😂 🙂 🙃 😉 😊 😇 🥰 😍 🤩 😎 🤔 🤗 🥳 😴 😭 😡 🤯 🫡",
    "animals": "🐶 🐱 🐭 🐹 🐰 🦊 🐻 🐼 🐨 🐯 🦁 🐮 🐸 🐵 🐧 🐦 🦋 🐢 🐬 🐳 🦄 🦖",
    "nature": "🌸 🌺 🌻 🌼 🌷 🌹 🪻 🌳 🌴 🍀 🌈 ☀️ 🌤️ 🌙 ⭐ 🌟 ⚡ 🔥 ❄️ 🌊",
    "food": "🍎 🍓 🍒 🍉 🍍 🥭 🍌 🍕 🍔 🍟 🌮 🍿 🍩 🍪 🎂 🍫 🍦 ☕ 🧃",
    "objects": "📱 💻 🎧 📷 💡 🔑 🎁 🎈 🎨 🚀 🛰️ ⚽ 🏆 🎮 📚 ✏️ 🔬 🔭",
    "symbols": "❤️ 🧡 💛 💚 💙 💜 🖤 🤍 💯 ✅ ❌ ⭐ ✨ ➡️ ⬆️ 🔔 ♻️ ™️ ®️",
}


def emoji_request_reply(message):
    """Return Unicode emoji locally, without waiting for the Gemini API."""
    lowered = message.lower().strip()
    request_words = (
        "emoji", "emojis", "send laughing", "smiling faces", "smiley faces",
        "heart emojis", "animal emojis", "food emojis", "send hearts",
        "give me hearts", "send some faces", "emoji art",
    )
    if not any(word in lowered for word in request_words):
        return None

    category_aliases = {
        "faces": ("face", "faces", "smile", "smiling", "laugh", "laughing", "sad", "angry"),
        "animals": ("animal", "animals", "dog", "cat", "bird", "fish"),
        "nature": ("nature", "flower", "flowers", "weather", "space", "plant"),
        "food": ("food", "fruit", "fruits", "pizza", "drink"),
        "objects": ("object", "objects", "technology", "tech", "rocket", "sports"),
        "symbols": ("heart", "hearts", "symbol", "symbols", "love", "check"),
    }
    category = next(
        (key for key, aliases in category_aliases.items()
         if any(alias in lowered for alias in aliases)),
        None,
    )
    if category:
        return EMOJI_SETS[category]
    # A mixed set is returned immediately; no AI request is needed.
    return " ".join(EMOJI_SETS.values())


# Small, reliable local answers for common factual questions. These bypass
# network/API latency; Gemini remains responsible for questions not listed here.
FAST_FACTS = {
    "speed of light": (
        "The speed of light in a vacuum is exactly 299,792,458 metres per second "
        "(m/s), approximately 3.00 × 10⁸ m/s. ⚡"
    ),
    "light speed": (
        "The speed of light in a vacuum is exactly 299,792,458 metres per second "
        "(m/s), approximately 3.00 × 10⁸ m/s. ⚡"
    ),
    "value of c": (
        "The speed of light constant, c, is exactly 299,792,458 m/s in a vacuum. ⚡"
    ),
}


def quick_fact_answer(message):
    normalized = re.sub(r"[^a-z0-9 ]", " ", (message or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    if any(phrase in normalized for phrase in (
        "speed of light", "light speed", "value of c", "how fast does light travel",
        "how fast is light", "how fast light travels",
    )):
        return FAST_FACTS["speed of light"]
    return None


def chat(message, image, history):
    message = (message or "").strip()
    history = history or []
    if not message and image is None:
        return history

    emoji_answer = emoji_request_reply(message) if image is None else None
    quick_answer = quick_math_answer(message) if image is None else None
    fact_answer = quick_fact_answer(message) if image is None else None
    instant_answer = (
        emoji_answer if emoji_answer is not None
        else quick_answer if quick_answer is not None
        else fact_answer
    )
    if instant_answer is not None:
        updated = list(history)
        updated.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": instant_answer},
        ])
        return updated

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

    user_parts = []
    if image is not None:
        try:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
            user_parts.append(types.Part.from_bytes(
                data=buffer.getvalue(), mime_type="image/jpeg"
            ))
        except Exception as exc:
            raise gr.Error(f"Could not prepare the uploaded picture: {exc}")
    user_parts.append(types.Part.from_text(
        text=message or "Please describe this picture and explain what it is showing."
    ))
    conversation.append(types.Content(role="user", parts=user_parts))

    try:
        answer = ask_gemini(conversation)
    except gr.Error as exc:
        answer = str(exc)

    updated = list(history)
    updated.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    return updated


def clear_chat():
    return [], ""


# ------------------------------
# Image generation (Cloudflare FLUX)
# ------------------------------

def generate_image(prompt):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter an image description.")
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error(
            "Cloudflare credentials are missing. Add CLOUDFLARE_API_TOKEN "
            "and CLOUDFLARE_ACCOUNT_ID in Render Environment."
        )

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            endpoint, headers=headers, json={"prompt": prompt[:2048], "steps": IMAGE_STEPS},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        # Cloudflare FLUX commonly returns JSON containing a base64 image.
        if "application/json" in content_type or response.content[:1] == b"{":
            try:
                data = response.json()
            except ValueError:
                raise gr.Error("Cloudflare returned invalid JSON.")

            result = data.get("result") or {}
            image_data = result.get("image") if isinstance(result, dict) else None
            if not image_data and isinstance(data, dict):
                image_data = data.get("image")

            if not image_data:
                detail = data.get("errors") or data.get("messages") or "No image field was returned."
                raise gr.Error(f"Cloudflare image generation failed: {detail}")

            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                image.load()
                return image.convert("RGB")
            except Exception as exc:
                logger.exception("Could not decode Cloudflare base64 image")
                raise gr.Error(f"Cloudflare returned image data that could not be decoded: {exc}")

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


# ------------------------------
# Image analysis
# ------------------------------

def analyze_image(image, question):
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88)
        image_part = types.Part.from_bytes(
            data=buffer.getvalue(), mime_type="image/jpeg"
        )
        prompt = (question or "").strip() or "Describe this image."
        return ask_gemini([image_part, prompt])
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise gr.Error(f"Image analysis failed: {exc}")


# ------------------------------
# Video analysis only
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


# ------------------------------
# Interface styling
# ------------------------------

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
footer {
    display: none !important;
}
.alpha-note {
    opacity: 0.75;
    font-size: 0.9rem;
}
"""

with gr.Blocks(title="Alpha AI") as app:
    gr.Markdown(
        """
        # ALPHA AI
        ### Think Beyond Limits.
        Chat, create images, analyze images and videos.
        """
    )

    with gr.Tab("AI Chat"):
        chatbot = gr.Chatbot(height=450, label="Alpha Assistant")
        chat_picture = gr.Image(
            type="pil", label="Attach a picture (optional)", height=180
        )
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Message Alpha or ask a question about your picture...",
                label="", show_label=False, scale=6, lines=1,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        gr.Markdown(
            "Ask questions, request emojis, or attach a picture and ask "
            "'What is this picture showing?' 📷",
            elem_classes=["alpha-note"],
        )
        send_button.click(
            fn=chat, inputs=[chat_input, chat_picture, chatbot], outputs=[chatbot]
        ).then(fn=lambda: "", inputs=None, outputs=[chat_input])
        chat_input.submit(
            fn=chat, inputs=[chat_input, chat_picture, chatbot], outputs=[chatbot]
        ).then(fn=lambda: "", inputs=None, outputs=[chat_input])
        clear_button = gr.Button("Clear chat")
        clear_button.click(fn=clear_chat, inputs=None, outputs=[chatbot, chat_input])

    with gr.Tab("Image Studio"):
        gr.Markdown("## Create images with Alpha")
        image_prompt = gr.Textbox(
            label="Image description",
            placeholder="Describe the image you want...",
        )
        image_button = gr.Button("Generate image", variant="primary")
        generated_image = gr.Image(label="Generated image", type="pil")
        image_button.click(
            fn=generate_image, inputs=[image_prompt], outputs=[generated_image]
        )

    with gr.Tab("Image Analysis"):
        uploaded_image = gr.Image(type="pil", label="Upload image")
        image_question = gr.Textbox(
            label="Your question", placeholder="What is in this image?"
        )
        analyze_image_button = gr.Button("Analyze image", variant="primary")
        image_result = gr.Textbox(label="Alpha's analysis", lines=8)
        analyze_image_button.click(
            fn=analyze_image,
            inputs=[uploaded_image, image_question],
            outputs=[image_result],
        )

    with gr.Tab("Video Analysis"):
        gr.Markdown(
            "## Analyze videos\n"
            "Upload a video and ask Alpha questions about its contents."
        )
        uploaded_video = gr.Video(label="Upload video")
        video_question = gr.Textbox(
            label="Your question", placeholder="Summarize this video..."
        )
        video_analyze_button = gr.Button("Analyze video", variant="primary")
        video_result = gr.Textbox(label="Video analysis", lines=8)
        video_analyze_button.click(
            fn=analyze_video,
            inputs=[uploaded_video, video_question],
            outputs=[video_result],
        )


# ------------------------------
# Render-compatible server start
# ------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        css=CSS,
        )
    
