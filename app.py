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

# ================ CONFIGURATION ================

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
You are Alpha AI, a helpful and accurate AI assistant.

ANSWER STRUCTURE:
1. Give the direct answer first.
2. Explain using numbered steps.
3. Show calculations when necessary.
4. Clearly identify the final answer.
5. Use simple English.
6. Keep answers focused and useful.

MATHEMATICS:
- Never use dollar signs as math delimiters.
- Never output raw LaTeX.
- Never use \\( \\) or \\[ \\].
- Use readable plain text.
- Use symbols such as ×, ÷, ², ≤ and ≥.
- Show the method before the final answer.

Do not invent facts. If a question is unclear, ask one
short clarification.
"""


# ================ FORMAT ANSWERS ================

def clean_answer(answer):

    if not answer:
        return "I couldn't generate an answer. Please try again."

    text = str(answer)

    text = text.replace("$$", "").replace("$", "")

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


# ================ LOCAL MATH ENGINE ================

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
                "Solution\n\n"
                "Final answer: "
                + str(sp.simplify(expression))
            )

        expanded = sp.expand(expression)

        if "factor" in q:

            factored = sp.factor(expanded)

            return (
                "Solution\n\n"
                "Step 1: Write the original expression.\n"
                f"{expanded}\n\n"
                "Step 2: Factorise the expression.\n\n"
                f"Final answer: {factored}"
            )

        if "solve" in q or "=" in question:

            roots = sp.solve(
                sp.Eq(expanded, 0), x
            )

            root_text = ", ".join(
                "x = " + str(root)
                for root in roots
            )

            return (
                "Solution\n\n"
                "Step 1: Set the expression equal to zero.\n"
                "Step 2: Solve the equation.\n\n"
                "Final answer: "
                + (root_text or "No roots found.")
            )

        if "expand" in q:
            return "Expanded form: " + str(expanded)

        if "simplify" in q:
            return (
                "Simplified form: "
                + str(sp.simplify(expression))
            )

        return (
            "Expression: " + str(expanded) + "\n\n"
            "Factorised form: " + str(sp.factor(expanded))
        )

    except Exception:
        return None


# ================ GEMINI CHAT ================

def ask_gemini(message, history):

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing in Render Environment."
        )

    contents = []

    for item in (history or [])[-8:]:

        if isinstance(item, dict) and item.get("role") in (
            "user", "assistant"
        ):

            contents.append(
                types.Content(
                    role=item["role"],
                    parts=[
                        types.Part(
                            text=str(item.get("content", ""))
                        )
                    ]
                )
            )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=1100
        )
    )

    return clean_answer(
        getattr(response, "text", "") or ""
    )


def chat_reply(message, history):

    message = (message or "").strip()
    history = history or []

    if not message:
        return history, history, ""

    try:

        # Local maths avoids an external API call.
        answer = local_math_answer(message)

        if answer is None:
            answer = ask_gemini(message, history)

        answer = clean_answer(answer)

    except Exception as error:

        log.exception("Chat error")

        answer = (
            "Alpha couldn't complete the request.\n"
            "Error: " + str(error)[:300]
        )

    updated = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    return updated, updated, ""


# ================ IMAGE GENERATION ================

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

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"
        },
        json={
            "prompt": prompt.strip(),
            "width": width,
            "height": height,
            "steps": 4
        },
        timeout=90
    )

    response.raise_for_status()

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
            raise gr.Error("Image provider returned no image.")

        image_bytes = base64.b64decode(encoded)

    output_path = (
        OUTPUT_DIR / f"alpha_{uuid.uuid4().hex}.png"
    )

    Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB").save(output_path, "PNG")

    return str(output_path)


# ================ IMAGE ANALYSIS ================

def analyze_image(image_path, question):

    if not image_path:
        raise gr.Error("Upload an image first.")

    if client is None:
        raise gr.Error("GEMINI_API_KEY is missing.")

    image = Image.open(image_path).convert("RGB")

    prompt = (question or "").strip()

    if not prompt:
        prompt = "Describe this image and its important details."

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=1100
        )
    )

    return clean_answer(
        getattr(response, "text", "") or ""
    )


# ================ VIDEO ANALYSIS ================

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
                "Summarise this video. Explain the main events "
                "using numbered steps."
            )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
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


# ================ USER INTERFACE ================

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

            chatbot = gr.Chatbot(height=450)
            chat_state = gr.State([])

            message = gr.Textbox(
                label="Message",
                placeholder="Ask Alpha anything...",
                lines=2
            )

            with gr.Row():

                send = gr.Button(
                    "Send", variant="primary"
                )

                clear = gr.Button("Clear conversation")

            send.click(
                chat_reply,
                [message, chat_state],
                [chatbot, chat_state, message]
            )

            message.submit(
                chat_reply,
                [message, chat_state],
                [chatbot, chat_state, message]
            )

            clear.click(
                lambda: ([], [], ""),
                None,
                [chatbot, chat_state, message]
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


# ================ START SERVER ================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=3
    ).launch(
        server_name="0.0.0.0",
        server_port=PORT
    )
    
