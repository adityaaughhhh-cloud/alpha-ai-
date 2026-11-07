import os
import io
import uuid
import time
import base64
import hashlib
import logging
import threading

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import requests

from PIL import Image, ImageOps


# ============================================================
# ALPHA AI 7.0 - IMAGE STUDIO
# IMAGE GENERATION ONLY
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ALPHA")


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLOUDFLARE_API_TOKEN = os.environ.get(
    "CLOUDFLARE_API_TOKEN"
)

CLOUDFLARE_ACCOUNT_ID = os.environ.get(
    "CLOUDFLARE_ACCOUNT_ID"
)

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", ""
).rstrip("/")

SUPABASE_SERVICE_ROLE_KEY = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY"
)

QUOTA_SALT = os.environ.get(
    "ALPHA_QUOTA_SALT"
)

SUPABASE_BUCKET = os.environ.get(
    "SUPABASE_BUCKET", "alpha-media"
)

DAILY_IMAGE_LIMIT = 10

IMAGE_TIMEOUT = 120

PORT = int(
    os.environ.get("PORT", "10000")
)

IST = ZoneInfo("Asia/Kolkata")

LOCAL_DIR = Path("/tmp/alpha-media")

LOCAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ENVIRONMENT VALIDATION
# ============================================================

required_variables = {

    "CLOUDFLARE_API_TOKEN":
        CLOUDFLARE_API_TOKEN,

    "CLOUDFLARE_ACCOUNT_ID":
        CLOUDFLARE_ACCOUNT_ID,

    "SUPABASE_URL":
        SUPABASE_URL,

    "SUPABASE_SERVICE_ROLE_KEY":
        SUPABASE_SERVICE_ROLE_KEY,

    "ALPHA_QUOTA_SALT":
        QUOTA_SALT

}

missing = [

    name

    for name, value in required_variables.items()

    if not value

]

if missing:

    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing)
    )

if len(QUOTA_SALT) < 32:

    raise RuntimeError(
        "ALPHA_QUOTA_SALT must contain "
        "at least 32 characters."
    )


# ============================================================
# 3. API CONFIGURATION
# ============================================================

http = requests.Session()

CLOUDFLARE_IMAGE_URL = (

    "https://api.cloudflare.com/client/v4/accounts/"

    + CLOUDFLARE_ACCOUNT_ID

    + "/ai/run/"

    + "@cf/black-forest-labs/flux-1-schnell"

)


# ============================================================
# 4. DAILY DATE
# ============================================================

def today():

    return datetime.now(IST).date().isoformat()


# ============================================================
# 5. ERROR HANDLING
# ============================================================

def get_error(error):

    logger.exception("ALPHA ERROR")

    message = str(error).lower()

    if "429" in message or "quota" in message:

        return (
            "Cloudflare API usage limit reached. "
            "Please try again later."
        )

    if "503" in message or "overloaded" in message:

        return (
            "Image generation service is busy. "
            "Please try again shortly."
        )

    if "401" in message:

        return "Cloudflare authentication failed."

    if "403" in message:

        return (
            "Cloudflare permission denied. "
            "Check your API token."
        )

    if "404" in message:

        return "Image generation model was not found."

    if "timeout" in message:

        return (
            "Image generation timed out. "
            "Please try again."
        )

    return (
        "Alpha AI could not generate your image. "
        "Please try again."
    )


# ============================================================
# 6. VISITOR IDENTIFICATION
# ============================================================

def visitor_hash(request):

    ip = "unknown"

    try:

        if request and request.client:

            ip = request.client.host or "unknown"

    except Exception:

        logger.warning(
            "Visitor identification failed."
        )

    return hashlib.sha256(

        (
            QUOTA_SALT + ":" + ip
        ).encode()

    ).hexdigest()


# ============================================================
# 7. SUPABASE HEADERS
# ============================================================

def supabase_headers():

    return {

        "apikey": SUPABASE_SERVICE_ROLE_KEY,

        "Authorization":
            "Bearer " + SUPABASE_SERVICE_ROLE_KEY,

        "Content-Type": "application/json"

    }


# ============================================================
# 8. CONSUME IMAGE QUOTA
# ============================================================

def consume_image_quota(visitor):

    endpoint = (

        SUPABASE_URL

        + "/rest/v1/rpc/consume_alpha_media_quota"

    )

    payload = {

        "p_visitor_hash": visitor,

        "p_usage_date": today(),

        "p_media_type": "image",

        "p_daily_limit": DAILY_IMAGE_LIMIT

    }

    response = requests.post(

        endpoint,

        headers=supabase_headers(),

        json=payload,

        timeout=20

    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):

        raise RuntimeError(
            "Invalid Supabase quota response."
        )

    if "allowed" not in result:

        raise RuntimeError(
            "Supabase quota function returned "
            "an unexpected response."
        )

    return result


# ============================================================
# 9. REFUND FAILED GENERATIONS
# ============================================================

def refund_image_quota(visitor, usage_date):

    endpoint = (

        SUPABASE_URL

        + "/rest/v1/rpc/refund_alpha_media_quota"

    )

    payload = {

        "p_visitor_hash": visitor,

        "p_usage_date": usage_date,

        "p_media_type": "image"

    }

    try:

        response = requests.post(

            endpoint,

            headers=supabase_headers(),

            json=payload,

            timeout=15

        )

        response.raise_for_status()

    except Exception:

        logger.exception(
            "IMAGE QUOTA REFUND FAILED"
        )


# ============================================================
# 10. IMAGE QUOTA STATUS
# ============================================================

def get_remaining_images(visitor):

    endpoint = (

        SUPABASE_URL

        + "/rest/v1/alpha_media_usage"

    )

    params = {

        "visitor_hash": "eq." + visitor,

        "usage_date": "eq." + today(),

        "media_type": "eq.image",

        "select": "used"

    }

    response = requests.get(

        endpoint,

        headers=supabase_headers(),

        params=params,

        timeout=15

    )

    response.raise_for_status()

    rows = response.json()

    used = rows[0]["used"] if rows else 0

    return max(
        0,
        DAILY_IMAGE_LIMIT - int(used)
    )


# ============================================================
# 11. IMAGE STYLES
# ============================================================

STYLE_GUIDANCE = {

    "Automatic": "",

    "Photorealistic":
        "Photorealistic appearance, realistic details, "
        "natural lighting.",

    "Cinematic":
        "Cinematic composition, dramatic lighting, "
        "professional film aesthetics.",

    "Anime":
        "High quality anime illustration.",

    "3D Render":
        "Professional detailed 3D render.",

    "Digital Art":
        "Detailed professional digital artwork.",

    "Watercolor":
        "Traditional watercolor painting.",

    "Oil Painting":
        "Detailed traditional oil painting.",

    "Pixel Art":
        "Detailed pixel art.",

    "Minimalist":
        "Clean minimalist artistic composition.",

    "Product Photography":
        "Professional studio product photography."

}


# ============================================================
# 12. IMAGE RATIOS
# ============================================================

RATIO_SIZES = {

    "1:1": (720, 720),

    "16:9": (1280, 720),

    "9:16": (720, 1280),

    "4:3": (960, 720),

    "3:4": (720, 960),

    "3:2": (1080, 720),

    "2:3": (720, 1080),

    "21:9": (1680, 720)

}


# ============================================================
# 13. PROMPT ENHANCEMENT
# ============================================================

def enhance_prompt(prompt, style, extra):

    guidance = STYLE_GUIDANCE.get(
        style,
        ""
    )

    final_prompt = (

        prompt.strip()

        + "\n"

        + guidance

        + "\n"

        + (extra or "")

    )

    return final_prompt[:2048]


# ============================================================
# 14. CLOUDFLARE IMAGE GENERATION
# ============================================================

def generate_flux_image(prompt):

    headers = {

        "Authorization":
            "Bearer " + CLOUDFLARE_API_TOKEN,

        "Content-Type":
            "application/json"

    }

    payload = {

        "prompt": prompt[:2048],

        "steps": 4

    }

    response = http.post(

        CLOUDFLARE_IMAGE_URL,

        headers=headers,

        json=payload,

        timeout=(10, IMAGE_TIMEOUT)

    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    # Handle JSON response.

    if "application/json" in content_type:

        data = response.json()

        if not data.get("success", False):

            errors = data.get("errors", [])

            raise RuntimeError(
                str(errors) or "Cloudflare generation failed."
            )

        encoded = data.get(
            "result", {}
        ).get("image")

        if not encoded:

            raise RuntimeError(
                "Cloudflare returned no image."
            )

        image_bytes = base64.b64decode(
            encoded
        )

    # Handle direct image response.

    else:

        image_bytes = response.content

    # Validate generated image.

    try:

        with Image.open(
            io.BytesIO(image_bytes)
        ) as image:

            image.verify()

    except Exception:

        raise RuntimeError(
            "Cloudflare returned invalid image data."
        )

    return image_bytes


# ============================================================
# 15. IMAGE EXPORT
# ============================================================

def export_image(data, ratio):

    width, height = RATIO_SIZES.get(
        ratio,
        (1280, 720)
    )

    with Image.open(
        io.BytesIO(data)
    ) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        result = ImageOps.pad(

            image,

            (width, height),

            method=Image.Resampling.LANCZOS,

            color=(15, 15, 20)

        )

        buffer = io.BytesIO()

        result.save(

            buffer,

            format="JPEG",

            quality=94

        )

        return buffer.getvalue(), result.copy()


# ============================================================
# 16. SUPABASE STORAGE
# ============================================================

def store_media(data, filename):

    endpoint = (

        SUPABASE_URL

        + "/storage/v1/object/"

        + SUPABASE_BUCKET

        + "/"

        + filename

    )

    headers = {

        "Authorization":
            "Bearer " + SUPABASE_SERVICE_ROLE_KEY,

        "apikey":
            SUPABASE_SERVICE_ROLE_KEY,

        "Content-Type":
            "image/jpeg",

        "x-upsert":
            "true"

    }

    try:

        response = requests.post(

            endpoint,

            headers=headers,

            data=data,

            timeout=(5, 30)

        )

        response.raise_for_status()

        logger.info(
            "Image stored: %s",
            filename
        )

    except Exception:

        logger.exception(
            "IMAGE STORAGE FAILED"
        )


def background_storage(data, filename):

    threading.Thread(

        target=store_media,

        args=(data, filename),

        daemon=True

    ).start()


# ============================================================
# 17. MAIN IMAGE GENERATOR
# ============================================================

def make_image(

    prompt,

    style,

    extra,

    ratio,

    request: gr.Request

):

    if not prompt or not prompt.strip():

        return (
            None,
            "Please describe the image.",
            None
        )

    if request is None:

        return (
            None,
            "Visitor session unavailable.",
            None
        )

    visitor = visitor_hash(request)

    usage_date = today()

    reserved = False

    started = time.monotonic()

    try:

        # Reserve one image credit.

        quota = consume_image_quota(
            visitor
        )

        if not quota.get("allowed"):

            return (

                None,

                """
                ### Daily image limit reached.

                You have used all 10 free images today.

                Your allowance resets at midnight India time.
                """,

                None

            )

        reserved = True

        # Prepare prompt.

        final_prompt = enhance_prompt(

            prompt,

            style,

            extra

        )

        # Generate image.

        raw_image = generate_flux_image(
            final_prompt
        )

        # Export image.

        final_bytes, display = export_image(

            raw_image,

            ratio

        )

        # Save locally.

        image_id = uuid.uuid4().hex

        output_path = (
            LOCAL_DIR / (image_id + ".jpg")
        )

        display.save(

            output_path,

            format="JPEG",

            quality=94

        )

        # Store image in Supabase.

        background_storage(

            final_bytes,

            "images/" + image_id + ".jpg"

        )

        elapsed = time.monotonic() - started

        remaining = quota.get(
            "remaining"
        )

        if remaining is None:

            remaining = get_remaining_images(
                visitor
            )

        status = f"""

# Image generated successfully!

**Alpha AI Image Studio**

- Engine: Cloudflare FLUX.1 Schnell
- Resolution: {display.width} × {display.height}
- Aspect ratio: {ratio}
- Generation time: {elapsed:.1f} seconds
- Remaining images today: {remaining}/10

Image ID: `{image_id}`

Your image is ready to download.

"""

        return (

            display,

            status,

            str(output_path)

        )

    except Exception as error:

        if reserved:

            refund_image_quota(

                visitor,

                usage_date

            )

        return (

            None,

            get_error(error),

            None

        )


# ============================================================
# 18. USER INTERFACE
# ============================================================

CSS = """

.gradio-container {

    max-width: 1150px !important;

    margin: auto !important;

}

#alpha-header {

    text-align: center;

    padding: 30px 10px;

}

#alpha-header h1 {

    font-size: 44px;

    font-weight: 900;

    letter-spacing: 3px;

}

#alpha-header p {

    font-size: 17px;

    opacity: .8;

}

button {

    border-radius: 12px !important;

}

textarea {

    border-radius: 12px !important;

}

"""


# ============================================================
# 19. ALPHA AI APPLICATION
# ============================================================

with gr.Blocks(

    title="ALPHA AI 7.0 - Image Studio"

) as app:

    gr.HTML("""

    <div id="alpha-header">

        <h1>🚀 ALPHA AI</h1>

        <p>THINK BEYOND LIMITS</p>

        <p>AI IMAGE STUDIO</p>

    </div>

    """)

    gr.Markdown("""

    ## Create images with Alpha AI

    Generate original images from your imagination.

    **10 free image generations per day.**

    """)

    prompt = gr.Textbox(

        label="Describe your image",

        placeholder=(
            "Example: A beautiful futuristic city "
            "on Mars at sunset..."
        ),

        lines=5,

        max_lines=10

    )

    with gr.Row():

        style = gr.Dropdown(

            choices=list(STYLE_GUIDANCE.keys()),

            value="Automatic",

            label="Image style"

        )

        ratio = gr.Dropdown(

            choices=list(RATIO_SIZES.keys()),

            value="16:9",

            label="Aspect ratio"

        )

    extra = gr.Textbox(

        label="Additional instructions",

        placeholder="Optional instructions...",

        lines=2

    )

    image_button = gr.Button(

        "🚀 Generate Image",

        variant="primary"

    )

    image_output = gr.Image(

        type="pil",

        label="Your generated image"

    )

    image_status = gr.Markdown()

    image_download = gr.File(

        label="Download your image"

    )

    image_button.click(

        fn=make_image,

        inputs=[

            prompt,

            style,

            extra,

            ratio

        ],

        outputs=[

            image_output,

            image_status,

            image_download

        ],

        api_name="generate_image"

    )

    gr.Markdown("""

    ---

    ## ALPHA AI

    **THINK BEYOND LIMITS**

    Powered by Cloudflare Workers AI.

    """)


# ============================================================
# 20. QUEUE
# ============================================================

app.queue(

    max_size=20,

    default_concurrency_limit=2

)


# ============================================================
# 21. START SERVER
# ============================================================

if __name__ == "__main__":

    logger.info(
        "STARTING ALPHA AI 7.0 IMAGE STUDIO"
    )

    app.launch(

        server_name="0.0.0.0",

        server_port=PORT,

        show_error=False,

        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        ),

        css=CSS

    )
