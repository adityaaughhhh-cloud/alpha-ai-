
import os
import re
import io
import time
import uuid
import base64
import logging
import threading
from pathlib import Path

import gradio as gr
import requests
import sympy as sp
from PIL import Image
from google import genai
from google.genai import types


# ================= CONFIGURATION =================

PORT = int(os.getenv("PORT", "10000"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-3.8-flash"
).strip()

GEMINI_BACKUP_MODEL = os.getenv(
    "GEMINI_BACKUP_MODEL", ""
).strip()

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN", ""
).strip()

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID", ""
).strip()

IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
).strip()

OUTPUT_DIR = Path("/tmp/alpha_ai_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_HISTORY = 6
MAX_OUTPUT_TOKENS = 800

GEMINI_SEMAPHORE = threading.BoundedSemaphore(1)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alpha")

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY else None
)


SYSTEM_PROMPT = """
You are Alpha AI, an intelligent and helpful AI assistant.

RULES:
1. Answer the actual question directly.
2. Explain difficult topics clearly.
3. Use simple English.
4. Give examples when useful.
5. Show calculations when necessary.
6. Avoid unnecessary introductions.
7. Never invent facts.
8. Explain uncertainty when necessary.
9. Provide useful answers without revealing hidden reasoning.
10. Give practical and accurate coding solutions.

MATHEMATICS:
- Use readable mathematics.
- Avoid raw LaTeX.
- Use ×, ÷, ², ≤, ≥ and π.
- Explain important calculation steps.

CODING:
- Give complete and practical solutions.
- Preserve existing functionality.
"""


# ================= ERROR HANDLING =================

def is_quota_error(error):
    msg = str(error).upper()

    return any(x in msg for x in [
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA_EXCEEDED"
    ])


def friendly_error(error):

    msg = str(error).upper()

    if is_quota_error(error):
        return (
            "### Alpha API limit reached\n\n"
            "Gemini temporarily refused this request.\n\n"
            "Please try again later or check your "
            "Google AI Studio quota and billing settings."
        )

    if "403" in msg or "PERMISSION_DENIED" in msg:
        return (
            "### API access denied\n\n"
            "Check your API key and model permissions."
        )

    if "404" in msg or "NOT_FOUND" in msg:
        return (
            "### Model unavailable\n\n"
            "Check your GEMINI_MODEL setting."
        )

    if "503" in msg or "UNAVAILABLE" in msg:
        return (
            "### Gemini is temporarily unavailable\n\n"
            "Please try again shortly."
        )

    log.error("Alpha error: %s", str(error)[:500])

    return (
        "Alpha couldn't complete your request. "
        "Please try again."
    )


# ================= ANSWER CLEANING =================

def clean_answer(answer):

    if not answer:
        return "I couldn't generate an answer."

    text = str(answer)

    text = text.replace("$$", "")
    text = re.sub(r"\\\[|\\\]", "", text)
    text = re.sub(r"\\\(|\\\)", "", text)

    replacements = {
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\rightarrow": "→",
        r"\pi": "π",
        r"\infty": "∞",
        r"\pm": "±"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        r"(\1)/(\2)",
        text
    )

    text = re.sub(
        r"\\text\{([^{}]*)\}",
        r"\1",
        text
    )

    text = text.replace(r"\left", "")
    text = text.replace(r"\right", "")

    return text.strip()


# ================= LOCAL MATH ENGINE =================

def local_math_answer(question):

    q = question.lower().strip()

    actions = (
        "calculate", "evaluate", "simplify",
        "expand", "factor", "factorise",
        "factorize", "solve", "what is"
    )

    if not any(word in q for word in actions):
        return None

    cleaned = re.sub(
        r"^(what is|calculate|evaluate|simplify|"
        r"expand|factor|factorise|factorize|solve|find)\s*",
        "",
        question.strip(),
        flags=re.IGNORECASE
    )

    cleaned = cleaned.strip(" ?:")
    cleaned = cleaned.replace("^", "**")
    cleaned = cleaned.replace("×", "*")
    cleaned = cleaned.replace("÷", "/")
    cleaned = re.sub(r"\s+", "", cleaned)

    if not cleaned or len(cleaned) > 100:
        return None

    if not re.fullmatch(r"[0-9xX+\-*/().=]+", cleaned):
        return None

    x = sp.Symbol("x")

    try:

        if "=" in cleaned:

            if cleaned.count("=") != 1:
                return None

            left, right = cleaned.split("=", 1)

            lhs = sp.sympify(left, locals={"x": x})
            rhs = sp.sympify(right, locals={"x": x})

            roots = sp.solve(
                sp.Eq(lhs, rhs), x
            )

            return "**Final answer:** " + (
                ", ".join("x = " + str(r) for r in roots)
                if roots else "No solution found."
            )

        expression = sp.sympify(
            cleaned,
            locals={"x": x}
        )

        if not expression.has(x):
            return (
                "**Final answer:** "
                + str(sp.simplify(expression))
            )

        expanded = sp.expand(expression)

        if "factor" in q:
            return (
                "**Factorised answer:** "
                + str(sp.factor(expanded))
            )

        if "expand" in q:
            return (
                "**Expanded answer:** "
                + str(expanded)
            )

        if "simplify" in q:
            return (
                "**Simplified answer:** "
                + str(sp.simplify(expression))
            )

        if "solve" in q:
            roots = sp.solve(
                sp.Eq(expanded, 0), x
            )

            return "**Answer:** " + str(roots)

        return None

    except Exception:
        return None


# ================= HISTORY =================

def prepare_history(history):

    contents = []

    valid = [
        item for item in (history or [])
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]

    for item in valid[-MAX_HISTORY:]:

        role = (
            "model"
            if item["role"] == "assistant"
            else "user"
        )

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(text=item["content"][-5000:])
                ]
            )
        )

    return contents


# ================= GEMINI STREAMING =================

def gemini_stream(model, contents):

    return client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.25,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )
    )


def stream_gemini_answer(message, history):

    if client is None:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    contents = prepare_history(history)

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
    )

    models = [GEMINI_MODEL]

    if (
        GEMINI_BACKUP_MODEL
        and GEMINI_BACKUP_MODEL != GEMINI_MODEL
    ):
        models.append(GEMINI_BACKUP_MODEL)

    for index, model in enumerate(models):

        accumulated = ""

        try:

            log.info("Gemini request: %s", model)

            with GEMINI_SEMAPHORE:

                stream = gemini_stream(model, contents)

                for chunk in stream:

                    chunk_text = getattr(
                        chunk, "text", None
                    )

                    if chunk_text:

                        accumulated += chunk_text

                        yield clean_answer(accumulated)

            if not accumulated.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return

        except Exception as error:

            log.exception("Gemini failed: %s", model)

            # Try the configured backup only if the
            # primary failed before producing an answer.
            if (
                is_quota_error(error)
                and not accumulated
                and index + 1 < len(models)
            ):
                continue

            raise


# ================= CHAT =================

def chat_reply(message, history):

    message = (message or "").strip()
    history = list(history or [])

    if not message:
        yield history, history, ""
        return

    working = history + [
        {"role": "user", "content": message}
    ]

    yield working, working, ""

    try:

        # Free local arithmetic without Gemini requests.
        answer = local_math_answer(message)

        if answer is not None:

            final = working + [
                {
                    "role": "assistant",
                    "content": clean_answer(answer)
                }
            ]

            yield final, final, ""
            return

        last_answer = ""

        for partial in stream_gemini_answer(
            message, history
        ):

            last_answer = partial

            live = working + [
                {
                    "role": "assistant",
                    "content": partial
                }
            ]

            yield live, live, ""

        final = working + [
            {
                "role": "assistant",
                "content": clean_answer(last_answer)
            }
        ]

        yield final, final, ""

    except Exception as error:

        log.exception("Chat error")

        final = working + [
            {
                "role": "assistant",
                "content": friendly_error(error)
            }
        ]

        yield final, final, ""


# ================= IMAGE GENERATION =================

def generate_image(prompt, ratio):

    prompt = (prompt or "").strip()

    if not prompt:
        raise gr.Error("Enter an image description.")

    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error("Cloudflare settings are missing.")

    dimensions = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1024, 768),
        "3:4": (768, 1024)
    }

    width, height = dimensions.get(
        ratio, (1024, 1024)
    )

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )

    try:

        response = requests.post(
            url,
            headers={
                "Authorization": (
                    f"Bearer {CLOUDFLARE_API_TOKEN}"
                )
            },
            json={
                "prompt": prompt,
                "width": width,
                "height": height
            },
            timeout=(15, 120)
        )

        if not response.ok:
            raise gr.Error(
                f"Image provider error: HTTP {response.status_code}"
            )

        if "image" in response.headers.get(
            "content-type", ""
        ).lower():

            image_bytes = response.content

        else:

            data = response.json()
            result = data.get("result") or {}

            encoded = (
                result.get("image")
                or result.get("data")
            )

            if isinstance(encoded, list):
                encoded = encoded[0] if encoded else None

            if not encoded:
                raise gr.Error(
                    "No image was returned."
                )

            if "," in encoded:
                encoded = encoded.split(",", 1)[1]

            image_bytes = base64.b64decode(encoded)

        output = (
            OUTPUT_DIR / f"alpha_{uuid.uuid4().hex}.png"
        )

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(output, "PNG")

        return str(output)

    except gr.Error:
        raise

    except Exception:
        log.exception("Image generation failed.")
        raise gr.Error(
            "Image generation failed. Check Cloudflare settings and usage."
        )


# ================= IMAGE ANALYSIS =================

def analyze_image(image_path, question):

    if not image_path:
        raise gr.Error("Upload an image first.")

    if client is None:
        raise gr.Error("GEMINI_API_KEY is missing.")

    try:

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        prompt = (question or "").strip()

        if not prompt:
            prompt = "Describe this image and its important details."

        with GEMINI_SEMAPHORE:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.25,
                    max_output_tokens=800
                )
            )

        return clean_answer(
            getattr(response, "text", "") or ""
        )

    except Exception as error:

        log.exception("Image analysis failed.")

        raise gr.Error(
            friendly_error(error)
        )


# ================= VIDEO ANALYSIS =================

def analyze_video(video_path, question):

    if not video_path:
        raise gr.Error("Upload a video first.")

    if client is None:
        raise gr.Error("GEMINI_API_KEY is missing.")

    uploaded = None

    try:

        uploaded = client.files.upload(
            file=video_path
        )

        deadline = time.time() + 180

        while (
            getattr(uploaded, "state", None)
            and str(uploaded.state).upper().endswith("PROCESSING")
        ):

            if time.time() > deadline:
                raise gr.Error(
                    "Video processing timed out."
                )

            time.sleep(2)

            uploaded = client.files.get(
                name=uploaded.name
            )

        if str(
            getattr(uploaded, "state", "")
        ).upper().endswith("FAILED"):

            raise gr.Error(
                "Video processing failed."
            )

        prompt = (question or "").strip()

        if not prompt:
            prompt = (
                "Summarise this video and explain its main events."
            )

        with GEMINI_SEMAPHORE:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[uploaded, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.25,
                    max_output_tokens=1000
                )
            )

        return clean_answer(
            getattr(response, "text", "") or ""
        )

    except gr.Error:
        raise

    except Exception as error:

        log.exception("Video analysis failed.")

        raise gr.Error(
            friendly_error(error)
        )

    finally:

        if uploaded is not None:

            try:
                client.files.delete(
                    name=uploaded.name
                )
            except Exception:
                log.warning("Video cleanup failed.")


# ================= USER INTERFACE =================

with gr.Blocks(
    title="ALPHA AI — Think Beyond Limits"
) as demo:

    gr.Markdown(
        "# ALPHA AI\n"
        "### Think Beyond Limits.\n"
        "Chat • Image Generation • Image Analysis • Video Analysis"
    )

    with gr.Tabs():

        # CHAT

        with gr.Tab("Chat"):

            chatbot = gr.Chatbot(
                height=450,
                type="messages"
            )

            chat_state = gr.State([])

            message = gr.Textbox(
                label="Message",
                placeholder="Ask Alpha anything...",
                lines=2
            )

            with gr.Row():

                send = gr.Button(
                    "Send",
                    variant="primary"
                )

                clear = gr.Button(
                    "Clear conversation"
                )

            send.click(
                chat_reply,
                inputs=[message, chat_state],
                outputs=[chatbot, chat_state, message]
            )

            message.submit(
                chat_reply,
                inputs=[message, chat_state],
                outputs=[chatbot, chat_state, message]
            )

            clear.click(
                lambda: ([], [], ""),
                outputs=[chatbot, chat_state, message],
                queue=False
            )

        # IMAGE STUDIO

        with gr.Tab("Image Studio"):

            prompt = gr.Textbox(
                label="Image description",
                placeholder="Describe the image...",
                lines=3
            )

            ratio = gr.Dropdown(
                choices=[
                    "1:1", "16:9", "9:16",
                    "4:3", "3:4"
                ],
                value="1:1",
                label="Image ratio"
            )

            generate = gr.Button(
                "Generate Image",
                variant="primary"
            )

            generated = gr.Image(
                label="Generated image",
                type="filepath"
            )

            generate.click(
                generate_image,
                inputs=[prompt, ratio],
                outputs=generated
            )

        # IMAGE ANALYSIS

        with gr.Tab("Image Analysis"):

            image_input = gr.Image(
                label="Upload image",
                type="filepath"
            )

            image_question = gr.Textbox(
                label="Question",
                placeholder="What should Alpha analyse?"
            )

            image_button = gr.Button(
                "Analyse Image",
                variant="primary"
            )

            image_output = gr.Markdown()

            image_button.click(
                analyze_image,
                inputs=[image_input, image_question],
                outputs=image_output
            )

        # VIDEO ANALYSIS

        with gr.Tab("Video Analysis"):

            video_input = gr.Video(
                label="Upload video"
            )

            video_question = gr.Textbox(
                label="Question",
                placeholder="What should Alpha analyse?"
            )

            video_button = gr.Button(
                "Analyse Video",
                variant="primary"
            )

            video_output = gr.Markdown()

            video_button.click(
                analyze_video,
                inputs=[video_input, video_question],
                outputs=video_output
            )

    gr.Markdown(
        "---\n**Alpha AI · Think Beyond Limits**"
    )


# ================= START SERVER =================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1,
        max_size=20
    ).launch(
        server_name="0.0.0.0",
        server_port=PORT
        )
            
