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
# ALPHA AI 8.0 - IMAGE ONLY
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ALPHA")

# ============================================================
# CONFIGURATION
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

DAILY_IMAGE_LIMIT = 10

IMAGE_TIMEOUT = 120

PORT = int(os.environ.get("PORT", "10000"))

IST = ZoneInfo("Asia/Kolkata")

OUTPUT_DIR = Path("/tmp/alpha-images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

required = {
    "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
    "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "ALPHA_QUOTA_SALT": QUOTA_SALT,
}

missing = [
    name for name, value in required.items()
    if not value
]

if missing:
    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing)
    )

if len(QUOTA_SALT) < 32:
    raise RuntimeError(
        "ALPHA_QUOTA_SALT must contain at least 32 characters."
    )

# ============================================================
# API CONNECTIONS
# ============================================================

http = requests.Session()

CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    + CLOUDFLARE_ACCOUNT_ID
    + "/ai/run/@cf/black-forest-labs/flux-1-schnell"
)

# ============================================================
# DATE
# ============================================================

def today():
    return datetime.now(IST).date().isoformat()


# ============================================================
# VISITOR IDENTIFICATION
# ============================================================

def get_visitor(request):

    ip = "unknown"

    try:
        if request and request.client:
            ip = request.client.host or "unknown"

    except Exception:
        logger.exception("VISITOR IDENTIFICATION ERROR")

    value = QUOTA_SALT + ":" + ip

    return hashlib.sha256(
        value.encode()
    ).hexdigest()


# ============================================================
# SUPABASE HEADERS
# ============================================================

def supabase_headers():

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": (
            "Bearer " + SUPABASE_SERVICE_ROLE_KEY
        ),
        "Content-Type": "application/json"
    }


# ============================================================
# RESERVE IMAGE CREDIT
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

    response = http.post(
        endpoint,
        headers=supabase_headers(),
        json=payload,
        timeout=20
    )

    if not response.ok:
        logger.error(
            "SUPABASE QUOTA ERROR: %s %s",
            response.status_code,
            response.text[:1000]
        )
        response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError("Invalid quota response")

    if "allowed" not in result:
        raise RuntimeError("Missing allowed field")

    return result


# ============================================================
# REFUND IMAGE CREDIT
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

        response = http.post(
            endpoint,
            headers=supabase_headers(),
            json=payload,
            timeout=20
        )

        if not response.ok:
            logger.error(
                "SUPABASE REFUND ERROR: %s",
                response.text[:1000]
            )

        response.raise_for_status()

    except Exception:
        logger.exception("IMAGE CREDIT REFUND FAILED")


# ============================================================
# IMAGE STYLES
# ============================================================

STYLES = {

    "Automatic": "",

    "Photorealistic":
        "Photorealistic, realistic lighting, natural details.",

    "Cinematic":
        "Cinematic composition, dramatic lighting, film quality.",

    "Anime":
        "High quality anime illustration.",

    "3D Render":
        "Professional detailed 3D render.",

    "Digital Art":
        "Professional digital artwork.",

    "Watercolor":
        "Traditional watercolor painting.",

    "Oil Painting":
        "Traditional detailed oil painting.",

    "Pixel Art":
        "High quality pixel art.",

    "Minimalist":
        "Clean minimalist artistic composition.",

    "Product Photography":
        "Professional studio product photography."

}


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

RATIOS = {

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
# PROMPT
# ============================================================

def build_prompt(prompt, style, extra):

    style_text = STYLES.get(style, "")

    final_prompt = (
        prompt.strip()
        + "\n"
        + style_text
        + "\n"
        + (extra or "")
    )

    return final_prompt[:2048]


# ============================================================
# CLOUDFLARE IMAGE GENERATION
# ============================================================

def generate_image(prompt):

    headers = {
        "Authorization": (
            "Bearer " + CLOUDFLARE_API_TOKEN
        ),
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "steps": 4
    }

    response = http.post(
        CLOUDFLARE_URL,
        headers=headers,
        json=payload,
        timeout=(10, IMAGE_TIMEOUT)
    )

    if not response.ok:
        logger.error(
            "CLOUDFLARE IMAGE ERROR: %s %s",
            response.status_code,
            response.text[:1000]
        )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "application/json" in content_type:

        data = response.json()

        if not data.get("success"):
            raise RuntimeError(
                "Cloudflare image generation failed"
            )

        encoded = data.get(
            "result", {}
        ).get("image")

        if not encoded:
            raise RuntimeError(
                "No image returned by Cloudflare"
            )

        image_bytes = base64.b64decode(encoded)

    else:

        image_bytes = response.content

    with Image.open(
        io.BytesIO(image_bytes)
    ) as image:

        image.verify()

    return image_bytes


# ============================================================
# IMAGE EXPORT
# ============================================================

def export_image(image_bytes, ratio):

    width, height = RATIOS.get(
        ratio,
        (1280, 720)
    )

    with Image.open(
        io.BytesIO(image_bytes)
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

        output = OUTPUT_DIR / (
            uuid.uuid4().hex + ".jpg"
        )

        result.save(
            output,
            format="JPEG",
            quality=94
        )

        return str(output)


# ============================================================
# MAIN GENERATOR
# ============================================================

def make_image(
    prompt,
    style,
    extra,
    ratio,
    request: gr.Request
):

    if not prompt or not prompt.strip():
        return None

    if request is None:
        return None

    visitor = get_visitor(request)

    usage_date = today()

    reserved = False

    try:

        # Check and reserve daily credit.

        quota = consume_image_quota(visitor)

        if not quota.get("allowed"):
            return None

        reserved = True

        # Build prompt.

        final_prompt = build_prompt(
            prompt,
            style,
            extra
        )

        # Generate.

        image_bytes = generate_image(
            final_prompt
        )

        # Export.

        output_path = export_image(
            image_bytes,
            ratio
        )

        return output_path

    except Exception:

        logger.exception(
            "ALPHA IMAGE GENERATION FAILED"
        )

        if reserved:

            refund_image_quota(
                visitor,
                usage_date
            )

        return None


# ============================================================
# DESIGN
# ============================================================

CSS = """

.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}

#header {
    text-align: center;
    padding: 25px 10px;
}

#header h1 {
    font-size: 46px;
    font-weight: 900;
    letter-spacing: 4px;
}

#header p {
    font-size: 16px;
    letter-spacing: 2px;
}

button {
    border-radius: 12px !important;
}

textarea {
    border-radius: 12px !important;
}

#result {
    border-radius: 16px;
    overflow: hidden;
}

"""


# ============================================================
# ALPHA AI INTERFACE
# ============================================================

with gr.Blocks(
    title="ALPHA AI - Image Studio"
) as app:

    gr.HTML("""
    <div id="header">
        <h1>🚀 ALPHA AI</h1>
        <p>THINK BEYOND LIMITS</p>
        <p>IMAGE STUDIO</p>
    </div>
    """)

    prompt = gr.Textbox(
        label="",
        placeholder="Describe the image you want to create...",
        lines=4,
        max_lines=8
    )

    with gr.Row():

        style = gr.Dropdown(
            choices=list(STYLES.keys()),
            value="Automatic",
            label="Style"
        )

        ratio = gr.Dropdown(
            choices=list(RATIOS.keys()),
            value="16:9",
            label="Aspect Ratio"
        )

    extra = gr.Textbox(
        label="Additional instructions",
        placeholder="Optional...",
        lines=2
    )

    generate = gr.Button(
        "🚀 Generate Image",
        variant="primary"
    )

    image_output = gr.Image(
        label="",
        type="filepath",
        show_download_button=True,
        show_share_button=False,
        interactive=False,
        elem_id="result"
    )

    generate.click(
        fn=make_image,
        inputs=[
            prompt,
            style,
            extra,
            ratio
        ],
        outputs=image_output,
        api_name="generate_image"
    )


# ============================================================
# QUEUE
# ============================================================

app.queue(
    max_size=20,
    default_concurrency_limit=2
)


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    logger.info(
        "ALPHA AI 8.0 IMAGE STUDIO STARTING"
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=False,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate"
        )
)
