import os
import io
import time
import logging
import threading
import re

from datetime import date

import gradio as gr

from PIL import Image
from google import genai
from google.genai import types


# =====================================================
# ALPHA AI 4.6
# CHAT + IMAGE ANALYSIS + VIDEO ANALYSIS
# CLEAN MATHEMATICAL FORMATTING
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ALPHA")


# =====================================================
# CONFIGURATION
# =====================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")


CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "gemini-3.5-flash-lite"
)

BACKUP_MODEL = os.environ.get(
    "BACKUP_MODEL",
    "gemini-3.1-flash-lite"
)


CHAT_HISTORY_LIMIT = 5

SIMPLE_TOKENS = 256
NORMAL_TOKENS = 1024
ADVANCED_TOKENS = 4096
MAX_OUTPUT_TOKENS = 4096

VIDEO_TIMEOUT = 100

PORT = int(
    os.environ.get("PORT", "10000")
)


client = genai.Client(
    api_key=API_KEY
)


# =====================================================
# SYSTEM INSTRUCTION
# =====================================================

SYSTEM_INSTRUCTION = """
You are Alpha AI, an intelligent AI assistant.

Name: Alpha AI
Tagline: Think Beyond Limits.

RULES:

1. Answer the actual question.
2. Be accurate and useful.
3. Never invent facts.
4. Keep simple answers short.
5. Explain complicated questions clearly.
6. Use simple English.
7. Avoid unnecessary introductions.
8. Do not repeat information.
9. Use conversation history when relevant.
10. Admit uncertainty when necessary.

DETAILED ANSWER RULES:

11. When the user asks for a derivation,
    provide the mathematical steps.

12. When equations are needed,
    explain important symbols.

13. Do not skip important calculation steps.

14. For step-by-step questions,
    organize the answer into numbered steps.

15. When the user asks for a detailed explanation,
    provide enough information.

16. Do not deliberately shorten detailed answers.

MATHEMATICAL FORMATTING RULES:

17. NEVER use LaTeX formatting.

18. NEVER use dollar signs ($) around equations.

19. NEVER output raw LaTeX commands.

20. Avoid commands such as:
    \\frac, \\text, \\sqrt, \\times, \\div.

21. Write mathematical equations in plain text.

22. Use normal mathematical symbols:
    +, -, ×, ÷, =, ≥, ≤.

23. Write chemical formulas clearly:
    H2O, NaOH, CO2, H2SO4.

24. Write units normally:
    g/mol, kg, m/s, joules.

25. Do not put unnecessary symbols
    around mathematical expressions.

26. Use readable numbered calculations.

27. Never output Markdown math delimiters.

Prioritize accuracy, clarity and useful answers.
"""


# =====================================================
# MATHEMATICAL FORMAT CLEANER
# =====================================================

def clean_math_format(text):

    if not text:
        return ""

    # Remove LaTeX display delimiters
    text = text.replace("$$", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")

    # Remove dollar signs
    text = text.replace("$", "")

    # Convert common mathematical commands
    replacements = {

        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pm": "±",
        r"\neq": "≠",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\approx": "≈",
        r"\pi": "π",
        r"\infty": "∞",
        r"\rightarrow": "→",
        r"\Rightarrow": "⇒",
        r"\degree": "°",
        r"\%": "%",
        r"\sqrt": "√",
        r"\alpha": "α",
        r"\beta": "β",
        r"\theta": "θ",
        r"\Delta": "Δ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\omega": "ω",
        r"\Omega": "Ω",
        r"\sigma": "σ",

    }

    for old, new in replacements.items():

        text = text.replace(old, new)

    # Convert simple fractions
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1)/(\2)",
        text
    )

    # Convert text commands
    text = re.sub(
        r"\\text\s*\{([^{}]*)\}",
        r"\1",
        text
    )

    text = re.sub(
        r"\\mathrm\s*\{([^{}]*)\}",
        r"\1",
        text
    )

    text = re.sub(
        r"\\mathbf\s*\{([^{}]*)\}",
        r"\1",
        text
    )

    # Remove remaining LaTeX spacing commands
    text = re.sub(
        r"\\(?:quad|qquad|,|;|!|:)",
        " ",
        text
    )

    # Remove remaining unnecessary braces
    text = text.replace("{", "")
    text = text.replace("}", "")

    # Remove unnecessary backslashes
    text = re.sub(
        r"\\([A-Za-z]+)",
        r"\1",
        text
    )

    # Clean excessive spaces
    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    return text.strip()


# =====================================================
# SMART TOKEN SELECTION
# =====================================================

def choose_token_limit(message):

    text = message.strip().lower()

    detailed_terms = [

        "derive",
        "derivation",
        "mathematically",
        "mathematical proof",
        "prove",
        "step by step",
        "step-by-step",
        "in detail",
        "detailed explanation",
        "full explanation",
        "show all steps",
        "with equations",
        "with examples",
        "comprehensive",
        "explain fully",
        "calculate step by step"

    ]

    if any(
        term in text
        for term in detailed_terms
    ):

        return ADVANCED_TOKENS

    word_count = len(text.split())

    if word_count <= 14 and len(text) <= 100:

        return SIMPLE_TOKENS

    if word_count <= 40 and len(text) <= 250:

        return NORMAL_TOKENS

    return MAX_OUTPUT_TOKENS


# =====================================================
# CHAT HISTORY
# =====================================================

def prepare_history(history):

    contents = []

    for item in (history or [])[-CHAT_HISTORY_LIMIT:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ["user", "assistant"]:
            continue

        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

        contents.append(

            types.Content(

                role="user" if role == "user" else "model",

                parts=[
                    types.Part.from_text(
                        text=content
                    )
                ]

            )

        )

    return contents


# =====================================================
# ERROR HANDLING
# =====================================================

def get_error(error):

    logger.exception(
        "ALPHA ERROR"
    )

    message = str(error).lower()

    if "429" in message or "quota" in message:

        return (
            "API quota reached. "
            "Please try again later."
        )

    if "503" in message or "overloaded" in message:

        return (
            "Alpha AI is experiencing high demand. "
            "Try again shortly."
        )

    if "404" in message:

        return (
            "The selected AI model is unavailable. "
            "Check the model configuration."
        )

    if "403" in message:

        return (
            "API access denied. "
            "Check Google API permissions."
        )

    if "timeout" in message:

        return (
            "The request took too long. "
            "Please try again."
        )

    if "api key" in message:

        return (
            "There is a problem with the Gemini API key."
        )

    return (
        "Alpha AI could not complete your request. "
        "Please try again."
    )


# =====================================================
# 1. AI CHAT
# =====================================================

def alpha_chat(message, history):

    if not message or not message.strip():

        yield "Please enter a message."
        return

    start = time.monotonic()

    message = message.strip()

    contents = prepare_history(history)

    contents.append(

        types.Content(

            role="user",

            parts=[
                types.Part.from_text(
                    text=message
                )
            ]

        )

    )

    token_limit = choose_token_limit(
        message
    )

    config = types.GenerateContentConfig(

        system_instruction=SYSTEM_INSTRUCTION,

        temperature=0.3,

        max_output_tokens=token_limit

    )

    models = [
        CHAT_MODEL,
        BACKUP_MODEL
    ]

    for index, model in enumerate(models):

        answer = ""

        try:

            logger.info(
                "CHAT START: model=%s tokens=%s",
                model,
                token_limit
            )

            stream = client.models.generate_content_stream(

                model=model,

                contents=contents,

                config=config

            )

            for chunk in stream:

                if chunk.text:

                    answer += chunk.text

                    yield clean_math_format(
                        answer
                    )

            if answer:

                elapsed = (
                    time.monotonic() - start
                )

                logger.info(
                    "CHAT COMPLETE: %.2f seconds",
                    elapsed
                )

                return

            yield "No response generated."
            return

        except Exception as error:

            logger.exception(
                "CHAT ERROR: %s",
                model
            )

            if answer:

                yield clean_math_format(
                    answer
                    + "\n\nConnection interrupted."
                )

                return

            if index == 0:

                logger.info(
                    "Trying backup model."
                )

                continue

            yield get_error(error)
            return


# =====================================================
# 2. IMAGE ANALYSIS
# =====================================================

def analyze_image(image_path, question):

    if not image_path:

        return "Please upload an image."

    if not question or not question.strip():

        question = (
            "Describe this image clearly."
        )

    start = time.monotonic()

    try:

        logger.info(
            "IMAGE ANALYSIS STARTED"
        )

        with Image.open(image_path) as source:

            image = source.convert("RGB")

            image.thumbnail(
                (1200, 1200)
            )

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=80
            )

            image_bytes = buffer.getvalue()

        contents = [

            types.Content(

                role="user",

                parts=[

                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    ),

                    types.Part.from_text(
                        text=question.strip()
                    )

                ]

            )

        ]

        config = types.GenerateContentConfig(

            system_instruction=SYSTEM_INSTRUCTION,

            temperature=0.3,

            max_output_tokens=1024

        )

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=contents,

            config=config

        )

        answer = clean_math_format(

            response.text
            or "No analysis generated."

        )

        elapsed = (
            time.monotonic() - start
        )

        logger.info(
            "IMAGE ANALYSIS COMPLETE: %.2f seconds",
            elapsed
        )

        return (

            answer

            + f"\n\nAnalysis time: {elapsed:.2f} seconds."

        )

    except Exception as error:

        return get_error(error)


# =====================================================
# VIDEO FILE STATE
# =====================================================

def get_file_state(video_file):

    state = getattr(
        video_file,
        "state",
        None
    )

    return getattr(
        state,
        "name",
        str(state)
    ).upper()


# =====================================================
# 3. VIDEO ANALYSIS
# =====================================================

def analyze_video(video_path, question):

    if not video_path:

        return "Please upload a video."

    if not question or not question.strip():

        question = (
            "Explain what happens in this video."
        )

    video_file = None

    start = time.monotonic()

    try:

        logger.info(
            "VIDEO UPLOAD STARTED"
        )

        video_file = client.files.upload(
            file=video_path
        )

        processing_start = time.monotonic()

        while True:

            video_file = client.files.get(
                name=video_file.name
            )

            state = get_file_state(
                video_file
            )

            if state == "ACTIVE":

                break

            if state == "FAILED":

                return (
                    "Video processing failed."
                )

            if (

                time.monotonic() - processing_start

                > VIDEO_TIMEOUT

            ):

                return (
                    "Video processing timed out."
                )

            time.sleep(3)

        contents = [

            types.Content(

                role="user",

                parts=[

                    types.Part.from_uri(

                        file_uri=video_file.uri,

                        mime_type=video_file.mime_type

                    ),

                    types.Part.from_text(

                        text=question.strip()

                    )

                ]

            )

        ]

        config = types.GenerateContentConfig(

            system_instruction=SYSTEM_INSTRUCTION,

            temperature=0.3,

            max_output_tokens=1024

        )

        response = client.models.generate_content(

            model=CHAT_MODEL,

            contents=contents,

            config=config

        )

        answer = clean_math_format(

            response.text

            or "No video analysis generated."

        )

        elapsed = (
            time.monotonic() - start
        )

        logger.info(
            "VIDEO ANALYSIS COMPLETE: %.2f seconds",
            elapsed
        )

        return (

            answer

            + f"\n\nAnalysis time: {elapsed:.2f} seconds."

        )

    except Exception as error:

        return get_error(error)

    finally:

        if video_file:

            try:

                client.files.delete(
                    name=video_file.name
                )

            except Exception:

                logger.warning(
                    "VIDEO CLEANUP FAILED"
                )


# =====================================================
# 4. DESIGN
# =====================================================

CSS = """

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

#alpha-header {
    text-align: center;
    padding: 18px;
}

#alpha-header h1 {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: 2px;
}

#alpha-header p {
    font-size: 14px;
    opacity: 0.7;
}

#chatbot {
    border-radius: 16px;
}

button {
    border-radius: 12px !important;
}

textarea {
    border-radius: 14px !important;
}

"""


# =====================================================
# 5. USER INTERFACE
# =====================================================

with gr.Blocks(
    title="ALPHA AI"
) as app:

    gr.HTML(

        """
        <div id="alpha-header">

            <h1>🚀 ALPHA AI</h1>

            <p>THINK BEYOND LIMITS</p>

            <p>Your intelligent AI assistant.</p>

        </div>
        """

    )

    with gr.Tabs():

        # CHAT

        with gr.Tab("💬 Chat"):

            gr.ChatInterface(

                fn=alpha_chat,

                title="Alpha AI",

                description=(

                    "Ask anything. "
                    "Get fast and intelligent answers."

                ),

                chatbot=gr.Chatbot(
                    height=530
                ),

                textbox=gr.Textbox(

                    placeholder="Message Alpha AI...",

                    container=True,

                    scale=7,

                    submit_btn="Send",

                    stop_btn="Stop"

                )

            )

        # IMAGE ANALYSIS

        with gr.Tab("🖼️ Image Analysis"):

            gr.Markdown(
                "## Understand any image"
            )

            image_input = gr.Image(

                label="Upload Image",

                type="filepath"

            )

            image_question = gr.Textbox(

                label="Ask Alpha",

                placeholder="What is in this image?"

            )

            image_analyze_button = gr.Button(

                "Analyze Image",

                variant="primary"

            )

            image_analysis_output = gr.Markdown()

            image_analyze_button.click(

                fn=analyze_image,

                inputs=[

                    image_input,

                    image_question

                ],

                outputs=image_analysis_output

            )

        # VIDEO ANALYSIS

        with gr.Tab("🎬 Video Analysis"):

            gr.Markdown(
                "## Understand Videos"
            )

            video_input = gr.Video(

                label="Upload Video",

                sources=["upload"]

            )

            video_question = gr.Textbox(

                label="Ask Alpha",

                placeholder=(

                    "Explain what happens in this video."

                )

            )

            video_analyze_button = gr.Button(

                "Analyze Video",

                variant="primary"

            )

            video_analysis_output = gr.Markdown()

            video_analyze_button.click(

                fn=analyze_video,

                inputs=[

                    video_input,

                    video_question

                ],

                outputs=video_analysis_output

            )

    gr.Markdown(

        """

        ---

        **ALPHA AI** | Think Beyond Limits

        Powered by Google Gemini.

        """

    )


# =====================================================
# 6. PERFORMANCE QUEUE
# =====================================================

app.queue(

    max_size=20,

    default_concurrency_limit=2

)


# =====================================================
# 7. START SERVER
# =====================================================

if __name__ == "__main__":

    logger.info(
        "STARTING ALPHA AI 4.6"
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        theme=gr.themes.Soft(

            primary_hue="blue",

            secondary_hue="slate"

        ),

        css=CSS,

        show_error=True

    )
