import os
import io
import time
import base64
import logging
import ast
import operator
import re
from functools import lru_cache

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types

# ============================================================
# ALPHA AI
# Faster streaming chat + improved image generation +
# image analysis + video analysis. No voice features.
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha-ai")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()

# Use valid, current Gemini API model IDs. Override in Render if needed.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite").strip()
_fallback_env = os.getenv(
    "CHAT_FALLBACK_MODELS",
    "gemini-3.8-flash,gemini-3.1-flash-lite"
)
CHAT_FALLBACK_MODELS = [
    model.strip() for model in _fallback_env.split(",") if model.strip()
]

# FLUX.2 [dev] is the higher-detail quality model.
# If the Cloudflare account cannot access it, set IMAGE_MODEL in Render to
# @cf/black-forest-labs/flux-1-schnell as a compatibility fallback.
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL", "@cf/black-forest-labs/flux-2-dev"
).strip()
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1024"))
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "28"))
IMAGE_GUIDANCE = float(os.getenv("IMAGE_GUIDANCE", "4.5"))

MAX_CHAT_HISTORY = 8
VIDEO_PROCESSING_TIMEOUT = 120
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def require_gemini():
    if gemini_client is None:
        raise gr.Error(
            "GEMINI_API_KEY is missing. Add it in Render > Environment, then redeploy."
        )
    return gemini_client


def clean_ai_text(text):
    if not text:
        return ""
    text = str(text)
    replacements = {
        r"\\rightarrow": " → ", r"\\to": " → ", r"\\times": " × ",
        r"\\div": " ÷ ", r"\\pm": " ± ", r"\\neq": " ≠ ",
        r"\\leq": " ≤ ", r"\\geq": " ≥ ", r"\\approx": " ≈ ",
        r"\\Delta": "Δ", r"\\alpha": "α", r"\\beta": "β",
        r"\\pi": "π", r"\\infty": "∞", r"\\cdot": " · ",
        r"\\degree": "°", r"\\%": "%", r"\\ ": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle)\b", "", text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace("$$", "").replace("$", "")
    sub_map = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    text = re.sub(
        r"_\{([0-9+\-=()]+)\}|_([0-9])",
        lambda m: (m.group(1) or m.group(2)).translate(sub_map), text
    )
    sup_map = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
    text = re.sub(
        r"\^\{([0-9+\-=()]+)\}|\^([0-9])",
        lambda m: (m.group(1) or m.group(2)).translate(sup_map), text
    )
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


SYSTEM_INSTRUCTION = (
    "You are Alpha, a fast, helpful and accurate assistant. Answer the actual "
    "request directly. Be clear and useful, not unnecessarily brief. For study "
    "routines, give a usable routine and briefly explain key choices. Use relevant "
    "Unicode emojis naturally. If asked for emojis, output emojis rather than "
    "describing them. Use readable Unicode symbols such as CO₂ and H₂O, not raw "
    "LaTeX delimiters. Show calculation steps when useful. If uncertain, say so."
)


def _model_list():
    result = []
    for model in [CHAT_MODEL, *CHAT_FALLBACK_MODELS]:
        if model and model not in result:
            result.append(model)
    return result


def ask_gemini(contents):
    """Non-streaming request used by image/video analysis."""
    client = require_gemini()
    last_error = None
    for model in _model_list():
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.4,
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            answer = clean_ai_text(getattr(response, "text", "") or "")
            if answer:
                logger.info("Request completed with %s", model)
                return answer
            last_error = RuntimeError(f"{model} returned an empty response")
        except Exception as exc:
            last_error = exc
            logger.warning("Model %s failed: %s", model, exc)
            msg = str(exc).lower()
            # Only route to another model for overload/rate-limit/model availability.
            retryable = any(term in msg for term in (
                "429", "500", "502", "503", "504", "unavailable",
                "resource exhausted", "high demand", "timeout", "timed out",
                "model not found", "not found", "unknown model", "unsupported",
            ))
            if not retryable:
                break
    logger.exception("Gemini request failed", exc_info=last_error)
    raise gr.Error(
        "Alpha could not get a response from Gemini. Check the API key, model access, "
        "quota and Render logs, then try again."
    )


def stream_gemini(contents):
    """Yield text chunks as soon as Gemini streams them; fallback if a model fails early."""
    client = require_gemini()
    last_error = None
    for model in _model_list():
        collected = []
        try:
            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.4,
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            for chunk in stream:
                piece = getattr(chunk, "text", None)
                if piece:
                    collected.append(piece)
                    yield clean_ai_text(piece), "".join(collected)
            if collected:
                logger.info("Streaming chat completed with %s", model)
                return
            last_error = RuntimeError(f"{model} returned no text")
        except Exception as exc:
            last_error = exc
            logger.warning("Streaming model %s failed: %s", model, exc)
            # Do not restart after visible text has already been sent.
            if collected:
                raise gr.Error("The response stream stopped. Please ask Alpha to continue.")
            msg = str(exc).lower()
            retryable = any(term in msg for term in (
                "429", "500", "502", "503", "504", "unavailable",
                "resource exhausted", "high demand", "timeout", "timed out",
                "model not found", "not found", "unknown model", "unsupported",
            ))
            if not retryable:
                break
    logger.error("All streaming models failed: %s", last_error)
    raise gr.Error(
        "Alpha's AI provider is unavailable. Check the Gemini API key, model access "
        "and quota, then retry."
    )


# ------------------------------
# Safe local calculator
# ------------------------------

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
    lowered = (message or "").lower().strip()
    if not any(word in lowered for word in (
        "emoji", "emojis", "send laughing", "smiling faces", "smiley faces",
        "heart emojis", "animal emojis", "food emojis", "send hearts",
        "give me hearts", "send some faces", "emoji art",
    )):
        return None
    aliases = {
        "faces": ("face", "faces", "smile", "smiling", "laugh", "laughing", "sad", "angry"),
        "animals": ("animal", "animals", "dog", "cat", "bird", "fish"),
        "nature": ("nature", "flower", "flowers", "weather", "space", "plant"),
        "food": ("food", "fruit", "fruits", "pizza", "drink"),
        "objects": ("object", "objects", "technology", "tech", "rocket", "sports"),
        "symbols": ("heart", "hearts", "symbol", "symbols", "love", "check"),
    }
    category = next(
        (key for key, words in aliases.items() if any(word in lowered for word in words)),
        None,
    )
    return EMOJI_SETS[category] if category else " ".join(EMOJI_SETS.values())


FAST_FACTS = {
    "speed of light": (
        "The speed of light in a vacuum is exactly 299,792,458 metres per second "
        "(m/s), approximately 3.00 × 10⁸ m/s. ⚡"
    ),
}


def quick_fact_answer(message):
    normalized = re.sub(r"[^a-z0-9 ]", " ", (message or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if any(phrase in normalized for phrase in (
        "speed of light", "light speed", "value of c", "how fast does light travel",
        "how fast is light", "how fast light travels",
    )):
        return FAST_FACTS["speed of light"]
    return None


def _history_to_contents(history, message, image=None):
    conversation = []
    for item in (history or [])[-MAX_CHAT_HISTORY:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            conversation.append(types.Content(
                role="user" if role == "user" else "model",
                parts=[types.Part.from_text(text=content)],
            ))
    parts = []
    if image is not None:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
        parts.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(
        text=message or "Please describe this picture and explain what it is showing."
    ))
    conversation.append(types.Content(role="user", parts=parts))
    return conversation


def chat(message, image, history):
    message = (message or "").strip()
    history = list(history or [])
    if not message and image is None:
        yield history
        return

    local_answer = None
    if image is None:
        local_answer = (
            emoji_request_reply(message)
            or quick_math_answer(message)
            or quick_fact_answer(message)
        )
    if local_answer is not None:
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": local_answer},
        ])
        yield history
        return

    try:
        contents = _history_to_contents(history, message, image)
    except Exception as exc:
        raise gr.Error(f"Could not prepare the request: {exc}")

    history.append({"role": "user", "content": message or "Please analyze this picture."})
    history.append({"role": "assistant", "content": ""})
    yield history

    full_answer = ""
    try:
        for piece, accumulated in stream_gemini(contents):
            full_answer = accumulated
            history[-1]["content"] = clean_ai_text(full_answer)
            yield history
    except gr.Error as exc:
        if not full_answer:
            history[-1]["content"] = str(exc)
            yield history
        else:
            history[-1]["content"] += "\n\n[Response interrupted. Ask Alpha to continue.]"
            yield history


def clear_chat():
    return [], ""


# ------------------------------
# Improved image generation
# ------------------------------

def improve_image_prompt(prompt):
    """Add concise visual-quality direction without changing the requested subject."""
    prompt = re.sub(r"\s+", " ", (prompt or "").strip())
    if not prompt:
        return ""
    # Do not force photorealism onto logos, icons, anime, diagrams or flat illustrations.
    non_photo = re.search(
        r"\b(logo|icon|sticker|anime|cartoon|illustration|vector|flat design|"
        r"infographic|diagram|pixel art|watercolor|sketch|comic)\b",
        prompt,
        re.I,
    )
    quality = (
        "Clear subject, intentional composition, accurate proportions, coherent "
        "details, balanced lighting, refined colors, crisp focus, professional "
        "visual quality. "
    )
    if non_photo:
        quality = (
            "Clean intentional composition, consistent visual style, precise shapes, "
            "coherent details, balanced colors, polished professional finish. "
        )
    return f"{prompt}. {quality}Avoid blurry details, distorted anatomy, duplicated objects, and visual clutter."


def _decode_cloudflare_image(response):
    content_type = response.headers.get("content-type", "").lower()
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
            detail = data.get("errors") or data.get("messages") or "No image was returned."
            raise gr.Error(f"Image generation failed: {detail}")
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


def generate_image(prompt):
    prompt = improve_image_prompt(prompt)
    if not prompt:
        raise gr.Error("Please enter an image description.")
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error(
            "Add CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in Render Environment."
        )

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt[:5000],
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "steps": IMAGE_STEPS,
        "guidance": IMAGE_GUIDANCE,
    }

    started = time.perf_counter()
    try:
        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
        )
        if not response.ok:
            detail = response.text[:700]
            logger.error("Cloudflare image API %s: %s", response.status_code, detail)
            if response.status_code in (400, 404, 422):
                raise gr.Error(
                    f"Cloudflare rejected IMAGE_MODEL '{IMAGE_MODEL}' or its settings. "
                    "Check model access and parameters in Render Environment. "
                    "For the old model, set IMAGE_MODEL to "
                    "@cf/black-forest-labs/flux-1-schnell and IMAGE_STEPS to 8."
                )
            response.raise_for_status()
        image = _decode_cloudflare_image(response)
        logger.info(
            "Image generated with %s in %.2fs", IMAGE_MODEL,
            time.perf_counter() - started
        )
        return image
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
# Video analysis
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
            raise gr.Error("Video processing timed out. Try a shorter video.")

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
# Interface
# ------------------------------

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
footer { display: none !important; }
.alpha-note { opacity: 0.75; font-size: 0.9rem; }
"""

with gr.Blocks(title="Alpha AI") as app:
    gr.Markdown(
        "# ALPHA AI\n### Think Beyond Limits.\n"
        "Chat, create images, analyze images and videos."
    )

    with gr.Tab("AI Chat"):
        chatbot = gr.Chatbot(height=450, label="Alpha Assistant", type="messages")
        chat_picture = gr.Image(
            type="pil", label="Attach a picture (optional)", height=180
        )
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Message Alpha or ask about your picture...",
                label="", show_label=False, scale=6, lines=1,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)
        gr.Markdown(
            "Answers stream as they are generated. Attach a picture and ask "
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
        gr.Markdown(
            "## Create images with Alpha\n"
            "Quality-focused generation uses FLUX.2 [dev] by default."
        )
        image_prompt = gr.Textbox(
            label="Image description",
            placeholder="Describe the subject, setting, style and mood...",
            lines=3,
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
            fn=analyze_image, inputs=[uploaded_image, image_question],
            outputs=[image_result],
        )

    with gr.Tab("Video Analysis"):
        gr.Markdown(
            "## Analyze videos\nUpload a video and ask Alpha questions about its contents."
        )
        uploaded_video = gr.Video(label="Upload video")
        video_question = gr.Textbox(
            label="Your question", placeholder="Summarize this video..."
        )
        video_analyze_button = gr.Button("Analyze video", variant="primary")
        video_result = gr.Textbox(label="Video analysis", lines=8)
        video_analyze_button.click(
            fn=analyze_video, inputs=[uploaded_video, video_question],
            outputs=[video_result],
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(),
        css=CSS,
    )
    
