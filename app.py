import os
import io
import re
import time
import uuid
import base64
import logging
import mimetypes
from pathlib import Path
from datetime import datetime, timezone

import gradio as gr
import requests
from PIL import Image, ImageOps
from google import genai
from google.genai import types


# ============================================================
# ALPHA AI 5.0
# Chat + Image Generation/Edit + Storage + Image Tools
# + Image Analysis + Video Analysis
#
# Storage:
# - Supabase Storage when configured (persistent)
# - Local temporary storage as a fallback (not persistent on Render)
#
# IMPORTANT:
# - Never put API keys in this file.
# - Configure secrets in Render Environment Variables.
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALPHA")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from environment variables.")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3.5-flash-lite")
BACKUP_MODEL = os.environ.get("BACKUP_MODEL", "gemini-3.1-flash-lite")

# Current Gemini image model IDs. Change these through Render env vars if needed.
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
IMAGE_PRO_MODEL = os.environ.get("IMAGE_PRO_MODEL", "gemini-3-pro-image")

VIDEO_TIMEOUT = int(os.environ.get("VIDEO_TIMEOUT", "100"))
PORT = int(os.environ.get("PORT", "10000"))
CHAT_HISTORY_LIMIT = 5

# Supabase Storage configuration. Keep the service-role key secret.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "alpha-images")

LOCAL_IMAGE_DIR = Path(os.environ.get("ALPHA_IMAGE_DIR", "/tmp/alpha-images"))
LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=API_KEY)


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are Alpha AI, an intelligent AI assistant.
Name: Alpha AI
Tagline: Think Beyond Limits.

Be accurate, useful, clear, and concise unless the user requests detail.
Never invent facts. Admit uncertainty. Use conversation history when relevant.
For derivations and calculations, show the important steps.
Use plain-text mathematical formatting; avoid raw LaTeX commands.
"""


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_math_format(text):
    if not text:
        return ""
    text = str(text)
    for old, new in {
        "$$": "", r"\(": "", r"\)": "", r"\[": "", r"\]": "",
        "$": "", r"\times": "×", r"\cdot": "·", r"\div": "÷",
        r"\pm": "±", r"\neq": "≠", r"\leq": "≤", r"\geq": "≥",
        r"\approx": "≈", r"\pi": "π", r"\infty": "∞",
        r"\rightarrow": "→", r"\Rightarrow": "⇒",
    }.items():
        text = text.replace(old, new)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def choose_token_limit(message):
    text = (message or "").strip().lower()
    detailed = (
        "derive", "derivation", "prove", "step by step", "step-by-step",
        "in detail", "detailed explanation", "full explanation",
        "show all steps", "with equations", "comprehensive"
    )
    if any(term in text for term in detailed):
        return 4096
    if len(text.split()) <= 14 and len(text) <= 100:
        return 256
    if len(text.split()) <= 40 and len(text) <= 250:
        return 1024
    return 4096


def prepare_history(history):
    contents = []
    for item in (history or [])[-CHAT_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        contents.append(types.Content(
            role="user" if role == "user" else "model",
            parts=[types.Part.from_text(text=content)]
        ))
    return contents


def get_error(error):
    logger.exception("ALPHA ERROR")
    message = str(error).lower()
    if "429" in message or "quota" in message or "resource_exhausted" in message:
        return "API quota reached. Please try again later or check your API billing and limits."
    if "503" in message or "overloaded" in message or "unavailable" in message:
        return "The AI service is busy. Please try again shortly."
    if "404" in message or "not_found" in message:
        return "The selected model or API endpoint is unavailable. Check the model setting."
    if "403" in message or "permission" in message:
        return "API access denied. Check API permissions and billing configuration."
    if "timeout" in message:
        return "The request took too long. Please try again."
    if "api key" in message or "unauthorized" in message:
        return "There is a problem with the Gemini API key."
    return "Alpha AI could not complete the request. Please try again."


# ============================================================
# 1. CHAT
# ============================================================

def alpha_chat(message, history):
    if not message or not message.strip():
        yield "Please enter a message."
        return

    contents = prepare_history(history)
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=message.strip())]
    ))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
        max_output_tokens=choose_token_limit(message)
    )

    for index, model in enumerate((CHAT_MODEL, BACKUP_MODEL)):
        answer = ""
        try:
            stream = client.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
            for chunk in stream:
                if chunk.text:
                    answer += chunk.text
                    yield clean_math_format(answer)
            if answer:
                return
            yield "No response generated."
            return
        except Exception as error:
            logger.exception("CHAT ERROR using %s", model)
            if answer:
                yield clean_math_format(answer + "\n\nConnection interrupted.")
                return
            if index == 1:
                yield get_error(error)
                return


# ============================================================
# 2. IMAGE ANALYSIS
# ============================================================

def image_to_bytes(image_path, max_side=1800):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        return buffer.getvalue(), "image/jpeg"


def analyze_image(image_path, question):
    if not image_path:
        return "Please upload an image."
    question = (question or "").strip() or "Describe this image clearly."
    try:
        image_bytes, mime = image_to_bytes(image_path)
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                types.Part.from_text(text=question)
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=1400
            )
        )
        return clean_math_format(response.text or "No analysis generated.")
    except Exception as error:
        return get_error(error)


# ============================================================
# 3. IMAGE PROMPT UNDERSTANDING + ENHANCEMENT
# ============================================================

STYLE_GUIDANCE = {
    "Automatic": "",
    "Photorealistic": "Photorealistic visual treatment, believable materials, natural details and coherent lighting.",
    "Cinematic": "Cinematic composition, intentional camera framing, atmospheric depth and controlled dramatic lighting.",
    "Anime": "High-quality anime illustration, expressive clean linework, coherent character design and polished cel shading.",
    "3D render": "Polished 3D-rendered look, convincing geometry, detailed materials, realistic reflections and studio-quality lighting.",
    "Digital art": "Detailed digital illustration with a deliberate composition, refined edges and a cohesive color palette.",
    "Watercolor": "Traditional watercolor appearance, translucent pigment, soft edges and visible paper texture.",
    "Oil painting": "Oil-painting appearance, layered brushwork, rich pigments and painterly texture.",
    "Pixel art": "Intentional pixel-art construction, crisp pixel clusters and a limited harmonious palette.",
    "Minimalist": "Minimalist composition, restrained details, clear visual hierarchy and generous negative space.",
    "Product photography": "Professional product-photography composition, accurate product shape, clean background and controlled studio lighting."
}


def understand_and_enhance_prompt(user_prompt, style, extra_instructions):
    """A lightweight prompt-structuring layer; it does not make a second AI API call."""
    prompt = (user_prompt or "").strip()
    if not prompt:
        raise ValueError("Enter a description of the image you want.")

    category = "general"
    lower = prompt.lower()
    categories = {
        "portrait": ("portrait", "person", "people", "face", "character"),
        "nature": ("forest", "mountain", "ocean", "river", "flower", "animal"),
        "space": ("galaxy", "planet", "space", "nebula", "astronaut", "star"),
        "architecture": ("building", "house", "architecture", "interior", "room"),
        "product": ("product", "bottle", "shoe", "watch", "phone", "mug"),
        "poster/design": ("poster", "banner", "logo", "advertisement", "flyer")
    }
    for name, words in categories.items():
        if any(word in lower for word in words):
            category = name
            break

    sections = [
        "Create one original image based on the user's request.",
        f"User's exact creative brief: {prompt}",
        f"Detected subject category (guidance only): {category}.",
        "Preserve the requested subject, objects, relationships, and important details.",
        "Use a coherent composition, clear subject separation, consistent perspective, "
        "believable lighting, intentional color, and carefully rendered details.",
        "Do not add unrelated objects, captions, logos, signatures, or watermarks unless requested."
    ]
    guidance = STYLE_GUIDANCE.get(style or "Automatic", "")
    if guidance:
        sections.append("Requested visual style: " + guidance)
    if extra_instructions and extra_instructions.strip():
        sections.append("Additional user instructions: " + extra_instructions.strip())
    return "\n".join(sections)


# ============================================================
# 4. IMAGE GENERATION + EDITING ENGINE
# ============================================================

def extract_generated_image(interaction):
    """Read the image block from the Gemini Interactions response."""
    output_image = getattr(interaction, "output_image", None)
    if output_image and getattr(output_image, "data", None):
        data = output_image.data
        if isinstance(data, str):
            return base64.b64decode(data)
        return data

    # Fallback for responses that expose image blocks through steps.
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", None) == "model_output":
            for block in getattr(step, "content", []) or []:
                if getattr(block, "type", None) == "image" and getattr(block, "data", None):
                    data = block.data
                    return base64.b64decode(data) if isinstance(data, str) else data
    raise RuntimeError("The image model returned no image. Try a different prompt or model.")


def save_generated_image(image_bytes, extension="png"):
    image_id = uuid.uuid4().hex
    local_path = LOCAL_IMAGE_DIR / f"{image_id}.{extension}"
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        image.save(local_path, format="PNG", optimize=True)
    return local_path, image_id


# ============================================================
# 5. PERSISTENT STORAGE ENGINE (SUPABASE STORAGE)
# ============================================================

def store_image(image_bytes, image_id, prompt, mime_type="image/png"):
    """
    Upload to a public Supabase Storage bucket.
    Configure SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and SUPABASE_BUCKET.
    The service-role key must remain private in Render environment variables.
    """
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        return None, "Local temporary copy only; persistent storage is not configured."

    safe_name = f"{datetime.now(timezone.utc):%Y/%m}/{image_id}.png"
    endpoint = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{safe_name}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": mime_type,
        "x-upsert": "true"
    }
    try:
        response = requests.post(endpoint, headers=headers, data=image_bytes, timeout=45)
        response.raise_for_status()
        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/"
            f"{SUPABASE_BUCKET}/{safe_name}"
        )
        return public_url, "Saved to persistent Supabase Storage."
    except Exception as error:
        logger.exception("SUPABASE STORAGE UPLOAD FAILED")
        return None, f"Image generated, but cloud storage upload failed: {error}"


def make_image_response(prompt, style, extra, aspect_ratio, image_size,
                        reference_image, do_2x_resize):
    start = time.monotonic()
    try:
        enhanced_prompt = understand_and_enhance_prompt(prompt, style, extra)

        # Pro model is selected for requested 4K output; Flash Image handles 1K/2K.
        model = IMAGE_PRO_MODEL if image_size == "4K" else IMAGE_MODEL
        response_format = {
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": aspect_ratio,
            "image_size": image_size
        }

        if reference_image:
            ref_bytes, ref_mime = image_to_bytes(reference_image, max_side=2500)
            interaction = client.interactions.create(
                model=model,
                input=[
                    {"type": "text", "text": (
                        enhanced_prompt +
                        "\nUse the supplied reference image as the visual starting point. "
                        "Apply the user's requested changes while preserving unrelated details."
                    )},
                    {
                        "type": "image",
                        "data": base64.b64encode(ref_bytes).decode("utf-8"),
                        "mime_type": ref_mime
                    }
                ],
                response_format=response_format
            )
        else:
            interaction = client.interactions.create(
                model=model,
                input=enhanced_prompt,
                response_format=response_format
            )

        image_bytes = extract_generated_image(interaction)

        # Optional high-quality 2x pixel resize. This is NOT AI super-resolution.
        if do_2x_resize:
            with Image.open(io.BytesIO(image_bytes)) as source:
                upscaled = source.convert("RGB").resize(
                    (source.width * 2, source.height * 2),
                    Image.Resampling.LANCZOS
                )
                buffer = io.BytesIO()
                upscaled.save(buffer, format="PNG", optimize=True)
                image_bytes = buffer.getvalue()

        local_path, image_id = save_generated_image(image_bytes)
        cloud_url, storage_message = store_image(
            image_bytes, image_id, prompt, "image/png"
        )

        with Image.open(local_path) as saved:
            display_image = saved.copy()

        elapsed = time.monotonic() - start
        details = [
            "### Alpha Image Generator",
            f"- **Model:** `{model}`",
            f"- **Requested size:** {image_size}",
            f"- **Aspect ratio:** {aspect_ratio}",
            f"- **Processing time:** {elapsed:.1f} seconds",
            f"- **Storage:** {storage_message}"
        ]
        if do_2x_resize:
            details.append("- **Resize:** 2× Lanczos resize (not AI super-resolution).")
        if cloud_url:
            details.append(f"- **Cloud image:** [Open saved image]({cloud_url})")
        details.append(f"- **Image ID:** `{image_id}`")
        return display_image, "\n".join(details)

    except Exception as error:
        return None, get_error(error)


# ============================================================
# 6. IMAGE UPSCALING / OPTIMIZATION TOOLS
# ============================================================

def resize_image_tool(image_path, scale):
    if not image_path:
        return None, "Upload an image first."
    try:
        factor = int(scale.replace("×", "").strip())
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            resized = image.resize(
                (image.width * factor, image.height * factor),
                Image.Resampling.LANCZOS
            )
            output = LOCAL_IMAGE_DIR / f"resize-{uuid.uuid4().hex}.png"
            resized.save(output, format="PNG", optimize=True)
            result = resized.copy()
        return result, (
            f"Resized to {resized.width} × {resized.height} pixels using Lanczos. "
            "This is high-quality resizing, not AI detail reconstruction."
        )
    except Exception as error:
        return None, get_error(error)


# ============================================================
# 7. VIDEO ANALYSIS (PRESERVED)
# ============================================================

def get_file_state(video_file):
    state = getattr(video_file, "state", None)
    return getattr(state, "name", str(state)).upper()


def analyze_video(video_path, question):
    if not video_path:
        return "Please upload a video."
    question = (question or "").strip() or "Explain what happens in this video."
    video_file = None
    start = time.monotonic()
    try:
        video_file = client.files.upload(file=video_path)
        processing_start = time.monotonic()
        while True:
            video_file = client.files.get(name=video_file.name)
            state = get_file_state(video_file)
            if state == "ACTIVE":
                break
            if state == "FAILED":
                return "Video processing failed."
            if time.monotonic() - processing_start > VIDEO_TIMEOUT:
                return "Video processing timed out."
            time.sleep(3)

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[
                types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type
                ),
                types.Part.from_text(text=question)
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                max_output_tokens=1400
            )
        )
        elapsed = time.monotonic() - start
        return clean_math_format(response.text or "No video analysis generated.") + (
            f"\n\nAnalysis time: {elapsed:.1f} seconds."
        )
    except Exception as error:
        return get_error(error)
    finally:
        if video_file:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                logger.warning("VIDEO FILE CLEANUP FAILED")


# ============================================================
# 8. UI
# ============================================================

CSS = """
.gradio-container { max-width: 1250px !important; margin: auto !important; }
#alpha-header { text-align: center; padding: 18px; }
#alpha-header h1 { font-size: 38px; font-weight: 800; letter-spacing: 2px; }
#alpha-header p { font-size: 14px; opacity: .75; }
button { border-radius: 12px !important; }
textarea { border-radius: 14px !important; }
"""

with gr.Blocks(title="ALPHA AI") as app:
    gr.HTML("""
    <div id="alpha-header">
      <h1>🚀 ALPHA AI</h1>
      <p>THINK BEYOND LIMITS</p>
      <p>Your intelligent AI assistant.</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            gr.ChatInterface(
                fn=alpha_chat,
                title="Alpha AI",
                description="Ask questions and get intelligent answers.",
                chatbot=gr.Chatbot(height=520),
                textbox=gr.Textbox(
                    placeholder="Message Alpha AI...",
                    submit_btn="Send",
                    stop_btn="Stop"
                )
            )

        with gr.Tab("🎨 Image Generator"):
            gr.Markdown(
                "## Create or edit images\n"
                "Describe the image. Optionally upload a reference image to edit."
            )
            prompt_input = gr.Textbox(
                label="What image should Alpha create?",
                placeholder="Example: A cinematic view of Saturn above an icy moon...",
                lines=4
            )
            with gr.Row():
                style_input = gr.Dropdown(
                    choices=list(STYLE_GUIDANCE.keys()),
                    value="Automatic",
                    label="Visual style"
                )
                ratio_input = gr.Dropdown(
                    choices=["1:1", "3:2", "2:3", "4:3", "3:4", "5:4",
                             "4:5", "16:9", "9:16", "21:9"],
                    value="1:1",
                    label="Aspect ratio"
                )
                size_input = gr.Dropdown(
                    choices=["1K", "2K", "4K"],
                    value="1K",
                    label="Output size"
                )
            extra_input = gr.Textbox(
                label="Optional extra instructions",
                placeholder="Colors, lighting, composition, details...",
                lines=2
            )
            reference_input = gr.Image(
                label="Optional reference image (for editing)",
                type="filepath"
            )
            resize_checkbox = gr.Checkbox(
                label="Apply optional 2× high-quality resize (not AI super-resolution)",
                value=False
            )
            generate_button = gr.Button("Generate image", variant="primary")
            generated_output = gr.Image(label="Generated image", type="pil")
            generation_status = gr.Markdown()
            generate_button.click(
                fn=make_image_response,
                inputs=[
                    prompt_input, style_input, extra_input, ratio_input,
                    size_input, reference_input, resize_checkbox
                ],
                outputs=[generated_output, generation_status]
            )

        with gr.Tab("🛠️ Image Tools"):
            gr.Markdown(
                "## Image resizing and optimization\n"
                "This tool enlarges pixels with Lanczos interpolation. It does not invent "
                "new details like a dedicated AI super-resolution model."
            )
            resize_input = gr.Image(label="Upload image", type="filepath")
            scale_input = gr.Dropdown(
                choices=["2×", "3×", "4×"],
                value="2×",
                label="Resize factor"
            )
            resize_button = gr.Button("Resize image", variant="primary")
            resized_output = gr.Image(label="Resized image", type="pil")
            resize_status = gr.Markdown()
            resize_button.click(
                fn=resize_image_tool,
                inputs=[resize_input, scale_input],
                outputs=[resized_output, resize_status]
            )

        with gr.Tab("🖼️ Image Analysis"):
            gr.Markdown("## Understand any image")
            image_input = gr.Image(label="Upload image", type="filepath")
            image_question = gr.Textbox(
                label="Ask Alpha",
                placeholder="What is in this image?"
            )
            image_analyze_button = gr.Button("Analyze image", variant="primary")
            image_analysis_output = gr.Markdown()
            image_analyze_button.click(
                fn=analyze_image,
                inputs=[image_input, image_question],
                outputs=image_analysis_output
            )

        with gr.Tab("🎬 Video Analysis"):
            gr.Markdown("## Understand videos")
            video_input = gr.Video(label="Upload video", sources=["upload"])
            video_question = gr.Textbox(
                label="Ask Alpha",
                placeholder="Explain what happens in this video."
            )
            video_analyze_button = gr.Button("Analyze video", variant="primary")
            video_analysis_output = gr.Markdown()
            video_analyze_button.click(
                fn=analyze_video,
                inputs=[video_input, video_question],
                outputs=video_analysis_output
            )

    gr.Markdown("---\n**ALPHA AI** | Think Beyond Limits | Powered by Google Gemini")

app.queue(max_size=20, default_concurrency_limit=2)

if __name__ == "__main__":
    logger.info("STARTING ALPHA AI 5.0")
    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CSS,
        show_error=True
    )
