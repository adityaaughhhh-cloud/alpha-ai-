import os
import gradio as gr
from google import genai
from google.genai import types

# Get API key from Hugging Face Secrets
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing")

# Initialize Gemini
client = genai.Client(api_key=api_key)

model_name = "gemini-3.5-flash-lite"

# Alpha AI engine with streaming
def alpha_chat(message, history):
    try:
        answer = ""

        stream = client.models.generate_content_stream(
            model=model_name,
            contents=message,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.7
            )
        )

        for chunk in stream:
            if chunk.text:
                answer += chunk.text
                yield answer

        if not answer:
            yield "Alpha couldn't generate a response."

    except Exception as e:
        yield f"ALPHA ERROR: {str(e)}"

# Chat interface
app = gr.ChatInterface(
    fn=alpha_chat,
    title="ALPHA AI",
    description="Think Beyond Limits"
)

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
