import os
import re
import io
import time
import uuid
import base64
import logging
from pathlib import Path

import gradio as gr
import requests
import sympy as sp
from PIL import Image
from google import genai
from google.genai import types

# ============== CONFIGURATION ==============

PORT = int(os.getenv("PORT", "10000"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-3.8-flash"
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alpha")

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY else None
)

SYSTEM_PROMPT = """
You are Alpha AI, an intelligent and helpful AI assistant.

ANSWER RULES:
1. Answer the user's actual question directly.
2. Explain difficult topics clearly and accurately.
3. Use numbered steps when they improve understanding.
4. Give examples when useful.
5. Show calculations when necessary.
6. Use simple English.
7. Avoid unnecessary introductions and repetition.
8. Never invent facts.
9. If uncertain, explain the uncertainty.
10. For difficult questions, think carefully before answering,
    but provide only the useful explanation, not hidden reasoning.

MATHEMATICS:
- Use plain text mathematics.
- Never use dollar signs as math delimiters.
- Never output raw LaTeX.
- Use ×, ÷, ², ≤, ≥ and π where appropriate.
- Explain important calculation steps.

CODING:
- Give practical, correct solutions.
- Preserve existing working functionality when modifying code.
"""

# ============== ANSWER CLEANING ==============

def clean_answer(answer):
    if not answer:
        return "I couldn't generate an answer. Please try again."

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
        r"\pm": "±",
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
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    return text.strip()


# ============== LOCAL MATH ENGINE ==============

def local_math_answer(question):

    q = question.lower().strip()

    actions = (
        "factor", "factorise", "factorize",
        "solve", "simplify", "expand",
        "calculate", "evaluate", "what is"
    )

    if not any(word in q for word in actions):
        return None

    cleaned = re.sub(
        r"^(what is|calculate|evaluate|simplify|expand|"
        r"factorise|factorize|factor|solve|find)\s*",
        "",
        question.strip(),
        flags=re.IGNORECASE
    )

    cleaned = cleaned.strip(" ?=:")
    cleaned = cleaned.replace("^", "**")
    cleaned = cleaned.replace("×", "*")
    cleaned = cleaned.replace("÷", "/")
    cleaned = re.sub(r"\s+", "", cleaned)

    if not cleaned or len(cleaned) > 100:
        return None

    if not re.fullmatch(r"[0-9xX+\-*/().*]+", cleaned):
        return None

    if not any(c.isdigit() for c in cleaned):
        return None

    x = sp.Symbol("x")

    try:
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
                "### Solution\n\n"
                f"Original expression: {expanded}\n\n"
                f"**Final answer:** {sp.factor(expanded)}"
            )

        if "solve" in q or "=" in question:
            roots = sp.solve(
                sp.Eq(expanded, 0), x
            )

            result = ", ".join(
                "x = " + str(root)
                for root in roots
            )

            return (
                "**Final answer:** "
                + (result or "No roots found.")
            )

        if "expand" in q:
            return "**Expanded form:** " + str(expanded)

        if "simplify" in q:
            return (
                "**Simplified form:** "
                + str(sp.simplify(expression))
            )

        return (
            "Expression: " + str(expanded)
            + "\n\nFactorised form: "
            + str(sp.factor(expanded))
        )

    except Exception:
        return None


# ============== GEMINI CHAT ==============

def prepare_history(history):

    contents = []

    # Keep only the last four exchanges.
    for item in (history or [])[-8:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content", "")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=content)]
            )
        )

    return contents


def stream_gemini_answer(message, history):

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing in Render Environment."
        )

    contents = prepare_history(history)

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
    )

    stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.25,
            max_output_tokens=1200
        )
    )

    accumulated = ""

    for chunk in stream:

        chunk_text = getattr(chunk, "text", None)

        if chunk_text:
            accumulated += chunk_text
            yield clean_answer(accumulated)

    if not accumulated.strip():
        yield "I couldn't generate an answer. Please try again."


# ============== CHAT RESPONSE ==============

def chat_reply(message, history):

    message = (message or "").strip()
    history = list(history or [])

    if not message:
        yield history, history, ""
        return

    working_history = history + [
        {"role": "user", "content": message}
    ]

    yield working_history, working_history, ""

    try:

        # Local mathematics first.
        answer = local_math_answer(message)

        if answer is not None:

            final_history = working_history + [
                {
                    "role": "assistant",
                    "content": clean_answer(answer)
                }
            ]

            yield final_history, final_history, ""
            return

        # Gemini streaming for general and difficult questions.
        last_answer = ""

        for partial_answer in stream_gemini_answer(
            message, history
        ):

            last_answer = partial_answer

            live_history = working_history + [
                {
                    "role": "assistant",
                    "content": partial_answer
                }
            ]

            yield live_history, live_history, ""

        final_history = working_history + [
            {
                "role": "assistant",
                "content": clean_answer(last_answer)
            }
        ]

        yield final_history, final_history, ""

    except Exception as error:

        log.exception("Chat error")

        error_message = (
            "Alpha couldn't complete your request.\n\n"
            "Error: " + str(error)[:250]
        )

        final_history = working_history + [
            {
                "role": "assistant",
                "content": error_message
            }
        ]

        yield final_history, final_history, ""


# ============== IMAGE GENERATION ==============

def generate_image(prompt, ratio):

    if not prompt or not prompt.strip():
        raise gr.Error("Enter an image description.")

    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error("Cloudflare API settings are missing.")

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
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"
            },
            json={
                "prompt": prompt.strip(),
                "width": width,
                "height": height
            },
            timeout=90
        )

        if not response.ok:
            raise gr.Error(
                f"Image provider error: {response.status_code} "
                + response.text[:400]
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
                    "Image provider returned no image."
                )

            image_bytes = base64.b64decode(encoded)

        output_path = (
            OUTPUT_DIR / f"alpha_{uuid.uuid4().hex}.png"
        )

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(output_path, "PNG")

        return str(output_path)

    except requests.RequestException as error:
        log.exception("Image generation error")
        raise gr.Error(str(error)[:300])


# ============== IMAGE ANALYSIS ==============

def analyze_image(image_path, question):

    if not image_path:
        raise gr.Error("Upload an image first.")

    if client is None:
        raise gr.Error("GEMINI_API_KEY is missing.")

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    prompt = (question or "").strip()

    if not prompt:
        prompt = "Describe this image and its important details."

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.25,
            max_output_tokens=1100
        )
    )

    return clean_answer(
        getattr(response, "text", "") or ""
    )


# ============== VIDEO ANALYSIS ==============

def analyze_video(video_path, question):

    if not video_path:
        raise gr.Error("Upload a video first.")

    if client is None:
        raise gr.Error("GEMINI_API_KEY is missing.")

    uploaded = None

    try:

        uploaded = client.files.upload(file=video_path)

        deadline = time.time() + 150

        while (
            getattr(uploaded, "state", None)
            and str(uploaded.state).upper().endswith("PROCESSING")
        ):

            if time.time() > deadline:
                raise gr.Error(
                    "Video processing timed out. Try a shorter video."
                )

            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        prompt = (question or "").strip()

        if not prompt:
            prompt = (
                "Summarise this video and explain its main events."
            )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.25,
                max_output_tokens=1300
            )
        )

        return clean_answer(
            getattr(response, "text", "") or ""
        )

    finally:

        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                log.warning("Temporary video cleanup failed.")


# ============== USER INTERFACE ==============

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

                clear = gr.Button("Clear conversation")

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
                inputs=None,
                outputs=[chatbot, chat_state, message],
                queue=False
            )

        # IMAGE STUDIO
        with gr.Tab("Image Studio"):

            prompt = gr.Textbox(
                label="Image description",
                placeholder="Describe the image you want...",
                lines=3
            )

            ratio = gr.Dropdown(
                choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
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
                [prompt, ratio],
                generated
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
                [image_input, image_question],
                image_output
            )

        # VIDEO ANALYSIS
        with gr.Tab("Video Analysis"):

            video_input = gr.Video(
                label="Upload video"
            )

            video_question = gr.Textbox(
                label="Question",
                placeholder="What should Alpha analyse in this video?"
            )

            video_button = gr.Button(
                "Analyse Video",
                variant="primary"
            )

            video_output = gr.Markdown()

            video_button.click(
                analyze_video,
                [video_input, video_question],
                video_output
            )

    gr.Markdown(
        "---\n**Alpha AI · Think Beyond Limits**"
    )


# ============== START SERVER ==============

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=3
    ).launch(
        server_name="0.0.0.0",
        server_port=PORT
                )
    
