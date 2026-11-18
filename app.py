
import os
import re
import ast
import time
import base64
import operator
import tempfile
import logging
from pathlib import Path

import gradio as gr
import requests
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha")

PORT = int(os.environ.get("PORT", "10000"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BACKUP_MODEL = os.environ.get(
    "GEMINI_BACKUP_MODEL",
    "gemini-3.5-flash-lite"
)

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

IMAGE_MODEL = os.environ.get(
    "IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
)

SYSTEM_PROMPT = """
You are Alpha, a helpful AI assistant.

Rules:
- Answer clearly and accurately.
- Use simple language when appropriate.
- Give the direct answer first.
- Avoid unnecessary introductions and repetition.
- For educational questions, explain step by step when useful.
- If you are uncertain, say so instead of inventing facts.
- Keep simple answers concise.
"""

# Create one reusable client instead of creating a new one
# for every question.
client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=45000)
        )
    except Exception:
        logger.exception("Could not initialize Gemini client")


# ============================================================
# LOCAL MATH ENGINE
# ============================================================

MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_math_node(node):
    if isinstance(node, ast.Expression):
        return safe_math_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        raise ValueError("Invalid number")

    if isinstance(node, ast.UnaryOp):
        operation = MATH_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Invalid operation")
        return operation(safe_math_node(node.operand))

    if isinstance(node, ast.BinOp):
        operation = MATH_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Invalid operation")

        left = safe_math_node(node.left)
        right = safe_math_node(node.right)

        # Prevent excessively large exponent calculations.
        if isinstance(node.op, ast.Pow):
            if abs(right) > 10 or abs(left) > 100000:
                raise ValueError("Calculation too large")

        result = operation(left, right)

        if isinstance(result, complex):
            raise ValueError("Invalid result")

        if abs(result) > 10**100:
            raise ValueError("Result too large")

        return result

    raise ValueError("Unsupported expression")


def extract_math_expression(message):
    text = message.strip().lower()

    # Direct expression, for example: 12 - 9
    if re.fullmatch(r"[\d\s.+\-*/()%]+", text):
        return text

    # Natural-language expression, for example:
    # "what is 12 - 9?"
    patterns = [
        r"^(?:what is|calculate|solve|compute)\s+(.+?)\??$",
        r"^(?:please calculate|please solve)\s+(.+?)\??$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            expression = match.group(1).strip()
            if re.fullmatch(r"[\d\s.+\-*/()%]+", expression):
                return expression

    return None


def calculate_locally(message):
    expression = extract_math_expression(message)

    if not expression:
        return None

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_math_node(tree)

        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return str(round(result, 10))

        return str(result)

    except Exception:
        return None


# ============================================================
# GEMINI GENERATION
# ============================================================

def check_client():
    if client is None:
        raise RuntimeError(
            "Gemini is not configured. Check GEMINI_API_KEY in Render."
        )


def is_temporary_error(error):
    message = str(error).upper()

    indicators = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "502",
        "504",
        "TIMEOUT",
        "INTERNAL"
    ]

    return any(item in message for item in indicators)


def generation_config():
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=900
    )


def generate_with_fallback(contents):
    """
    Try the primary model first.
    Use the backup model if the primary has a temporary failure.
    """

    check_client()

    models = [GEMINI_MODEL]

    if GEMINI_BACKUP_MODEL and GEMINI_BACKUP_MODEL != GEMINI_MODEL:
        models.append(GEMINI_BACKUP_MODEL)

    last_error = None

    for index, model_name in enumerate(models):
        try:
            logger.info("Trying Gemini model: %s", model_name)

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config()
            )

            if response and response.text:
                return response.text

            raise RuntimeError("Gemini returned an empty response.")

        except Exception as error:
            last_error = error
            logger.warning(
                "Model %s failed: %s",
                model_name,
                str(error)[:300]
            )

            if not is_temporary_error(error):
                break

            if index < len(models) - 1:
                time.sleep(0.3)

    raise RuntimeError(
        "Alpha could not get a response from Gemini. "
        "Please try again shortly. Details: "
        + str(last_error)[:300]
    )


def stream_with_fallback(contents):
    """
    Stream the answer from the primary model.
    If the stream fails before producing text, try the backup.
    """

    check_client()

    models = [GEMINI_MODEL]

    if GEMINI_BACKUP_MODEL and GEMINI_BACKUP_MODEL != GEMINI_MODEL:
        models.append(GEMINI_BACKUP_MODEL)

    last_error = None

    for index, model_name in enumerate(models):
        try:
            logger.info("Starting stream with: %s", model_name)

            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generation_config()
            )

            received_text = False

            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)

                if chunk_text:
                    received_text = True
                    yield chunk_text

            if not received_text:
                raise RuntimeError("Empty streaming response.")

            return

        except Exception as error:
            last_error = error

            logger.warning(
                "Streaming failed for %s: %s",
                model_name,
                str(error)[:300]
            )

            # Do not restart from the backup after showing
            # part of an answer, because that could duplicate text.
            if received_text:
                raise RuntimeError(
                    "The response stream stopped unexpectedly. "
                    "Please try again."
                )

            if not is_temporary_error(error):
                break

            if index < len(models) - 1:
                time.sleep(0.3)

    raise RuntimeError(
        "Alpha is temporarily unable to answer. "
        "Please try again shortly. Details: "
        + str(last_error)[:300]
    )


# ============================================================
# CHAT
# ============================================================

def convert_history(history, current_message):
    contents = []

    for item in (history or [])[-8:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        contents.append(
            types.Content(
                role="user" if role == "user" else "model",
                parts=[types.Part(text=content)]
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=current_message)]
        )
    )

    return contents


def chat(message, history):
    history = list(history or [])

    if not message or not message.strip():
        yield history, ""
        return

    message = message.strip()

    # FIRST: Try local calculation.
    # This avoids a Gemini API request for simple arithmetic.
    local_answer = calculate_locally(message)

    if local_answer is not None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": local_answer})
        yield history, ""
        return

    history.append({"role": "user", "content": message})

    try:
        contents = convert_history(history[:-1], message)

        # Display an empty assistant message, then update it
        # as Gemini streams tokens.
        history.append({"role": "assistant", "content": ""})

        answer = ""

        for part in stream_with_fallback(contents):
            answer += part
            history[-1]["content"] = answer
            yield history, ""

    except Exception as error:
        logger.exception("Chat error")

        error_message = (
            "Sorry, Alpha could not complete this request. "
            + str(error)
        )

        if history and history[-1].get("role") == "assistant":
            history[-1]["content"] = error_message
        else:
            history.append({
                "role": "assistant",
                "content": error_message
            })

        yield history, ""


# ============================================================
# IMAGE GENERATION - CLOUDFLARE FLUX
# ============================================================

def generate_image(prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter an image prompt.")

    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        raise gr.Error(
            "Image generation is not configured. "
            "Check your Cloudflare environment variables."
        )

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + CLOUDFLARE_ACCOUNT_ID
        + "/ai/run/"
        + IMAGE_MODEL
    )

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt.strip(),
        "num_steps": 4
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code != 200:
            raise gr.Error(
                "Image provider error: "
                + response.text[:500]
            )

        content_type = response.headers.get("content-type", "")

        if "image/" in content_type:
            image_bytes = response.content
        else:
            data = response.json()

            if not data.get("success", True):
                raise gr.Error(str(data.get("errors", data))[:500])

            result = data.get("result", {})
            encoded_image = result.get("image")

            if not encoded_image:
                raise gr.Error("The image provider returned no image.")

            image_bytes = base64.b64decode(encoded_image)

        output_path = os.path.join(
            tempfile.gettempdir(),
            f"alpha_image_{int(time.time() * 1000)}.png"
        )

        with open(output_path, "wb") as file:
            file.write(image_bytes)

        # Verify that the returned file is a valid image.
        with Image.open(output_path) as generated:
            generated.verify()

        return output_path

    except gr.Error:
        raise
    except Exception as error:
        logger.exception("Image generation error")
        raise gr.Error("Image generation failed: " + str(error)[:300])


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image, question):
    if image is None:
        raise gr.Error("Please upload an image.")

    prompt = (question or "").strip()

    if not prompt:
        prompt = "Describe this image clearly and accurately."

    try:
        contents = [
            image,
            prompt
        ]

        return generate_with_fallback(contents)

    except Exception as error:
        logger.exception("Image analysis error")
        raise gr.Error("Image analysis failed: " + str(error)[:400])


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path, question):
    if not video_path:
        raise gr.Error("Please upload a video.")

    prompt = (question or "").strip()

    if not prompt:
        prompt = "Summarize this video and explain what happens."

    try:
        check_client()

        uploaded_file = client.files.upload(file=video_path)

        # Wait for video processing when the API reports
        # that the uploaded file is still processing.
        for _ in range(30):
            state = getattr(uploaded_file, "state", None)
            state_name = str(state).upper()

            if "ACTIVE" in state_name:
                break

            if "FAILED" in state_name:
                raise RuntimeError("Google could not process this video.")

            time.sleep(1)

            uploaded_file = client.files.get(
                name=uploaded_file.name
            )

        contents = [
            uploaded_file,
            prompt
        ]

        return generate_with_fallback(contents)

    except Exception as error:
        logger.exception("Video analysis error")
        raise gr.Error("Video analysis failed: " + str(error)[:400])


# ============================================================
# GRADIO INTERFACE
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    title="Alpha AI",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS
) as demo:

    gr.Markdown(
        """
        # Alpha AI
        ### Think Beyond Limits.
        """
    )

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            height=520,
            label="Alpha Chat"
        )

        with gr.Row():
            message = gr.Textbox(
                placeholder="Ask Alpha anything...",
                show_label=False,
                scale=8,
                lines=1
            )

            send_button = gr.Button(
                "Send",
                variant="primary",
                scale=1
            )

        clear_button = gr.Button("Clear chat")

        message.submit(
            chat,
            inputs=[message, chatbot],
            outputs=[chatbot, message]
        )

        send_button.click(
            chat,
            inputs=[message, chatbot],
            outputs=[chatbot, message]
        )

        clear_button.click(
            lambda: ([], ""),
            outputs=[chatbot, message]
        )

    with gr.Tab("Image Studio"):
        gr.Markdown("### Create an image with Alpha")

        image_prompt = gr.Textbox(
            label="Image prompt",
            placeholder="Describe the image you want..."
        )

        generate_button = gr.Button(
            "Generate Image",
            variant="primary"
        )

        generated_image = gr.Image(
            label="Generated image",
            type="filepath"
        )

        download_file = gr.File(
            label="Download generated image"
        )

        generate_button.click(
            generate_image,
            inputs=image_prompt,
            outputs=[generated_image, download_file]
        ).then(
            lambda path: path,
            inputs=generated_image,
            outputs=download_file
        )

    with gr.Tab("Image Analysis"):
        gr.Markdown("### Ask questions about an image")

        uploaded_image = gr.Image(
            label="Upload image",
            type="pil"
        )

        image_question = gr.Textbox(
            label="Your question",
            placeholder="What is happening in this image?"
        )

        analyze_image_button = gr.Button(
            "Analyze Image",
            variant="primary"
        )

        image_result = gr.Markdown()

        analyze_image_button.click(
            analyze_image,
            inputs=[uploaded_image, image_question],
            outputs=image_result
        )

    with gr.Tab("Video Analysis"):
        gr.Markdown("### Understand a video with Alpha")

        uploaded_video = gr.Video(
            label="Upload video"
        )

        video_question = gr.Textbox(
            label="Your question",
            placeholder="Summarize this video..."
        )

        analyze_video_button = gr.Button(
            "Analyze Video",
            variant="primary"
        )

        video_result = gr.Markdown()

        analyze_video_button.click(
            analyze_video,
            inputs=[uploaded_video, video_question],
            outputs=video_result
        )

    gr.Markdown(
        """
        ---
        **Alpha AI** · Think Beyond Limits.
        """
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=4,
        max_size=30
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT
                   )
    
