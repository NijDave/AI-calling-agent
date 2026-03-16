import datetime
import json
import os
import time

import pytz
import requests
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from flask_cors import CORS
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai").strip().lower()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "marin")
OPENAI_REALTIME_TRANSCRIBE_MODEL = os.getenv(
    "OPENAI_REALTIME_TRANSCRIBE_MODEL",
    "gpt-4o-mini-transcribe",
)
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    (
        "Speak naturally like a friendly assistant. Keep the delivery clear, smooth, "
        "and natural. If the text is in Hindi or Hinglish, pronounce it fluently "
        "and naturally."
    ),
)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
ELEVENLABS_LATENCY_MODE = int(os.getenv("ELEVENLABS_LATENCY_MODE", "3"))

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
speak_request_cache = {}

SYSTEM_PROMPT = (
    "You are a warm, natural, friendly AI assistant having a real conversation. "
    "Reply like a human would speak in daily life: smooth, clear, and relaxed. "
    "Do not sound robotic, formal, scripted, or overly polished. "
    "Keep replies short, usually 1 to 2 sentences, unless the user clearly asks for more. "
    "Match the user's language naturally. "
    "If the user speaks in Hindi or Hinglish, reply naturally in Hindi or Hinglish. "
    "If the user speaks in English, reply naturally in English. "
    "Use contractions in English when they sound natural. "
    "Avoid lists unless the user asks for them. "
    "If asked for time or date, use the provided India time exactly. "
    "If unsure about a fact, say so briefly instead of making something up."
)


def preferred_tts_provider():
    if (
        TTS_PROVIDER == "elevenlabs"
        and elevenlabs_client
        and ELEVENLABS_VOICE_ID
    ):
        return "elevenlabs"
    return "openai"


def require_openai_client():
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return openai_client


def build_system_prompt():
    india_time = datetime.datetime.now(
        pytz.timezone("Asia/Kolkata")
    ).strftime("%I:%M %p, %A %d %B %Y")
    return (
        f"{SYSTEM_PROMPT} Current India date and time is {india_time}. "
        "Never say you cannot access the time."
    )


def build_realtime_session_config():
    return {
        "type": "realtime",
        "model": OPENAI_REALTIME_MODEL,
        "instructions": build_system_prompt(),
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "noise_reduction": {
                    "type": "near_field",
                },
                "transcription": {
                    "model": OPENAI_REALTIME_TRANSCRIBE_MODEL,
                    "prompt": (
                        "Expect Indian English, Hindi, and Hinglish code-switching. "
                        "Preserve mixed-language phrases and names naturally."
                    ),
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.45,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 550,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {
                "voice": OPENAI_REALTIME_VOICE,
                "speed": 0.96,
            },
        },
    }


def build_realtime_client_secret_payload():
    return {
        "expires_after": {
            "anchor": "created_at",
            "seconds": 60,
        },
        "session": build_realtime_session_config(),
    }


def get_speak_text():
    if request.method == "GET":
        return request.args.get("text", "").strip()
    data = request.get_json(silent=True) or {}
    return data.get("text", "").strip()


def is_duplicate_speak_request():
    request_token = request.args.get("t", "").strip()
    if not request_token:
        return False

    now = time.time()
    expired_tokens = [
        token for token, seen_at in speak_request_cache.items() if now - seen_at > 15
    ]
    for token in expired_tokens:
        speak_request_cache.pop(token, None)

    if request_token in speak_request_cache:
        return True

    speak_request_cache[request_token] = now
    return False


def stream_openai_audio(text):
    client = require_openai_client()
    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        instructions=OPENAI_TTS_INSTRUCTIONS,
        response_format="mp3",
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            yield chunk


def stream_elevenlabs_audio(text):
    if not elevenlabs_client or not ELEVENLABS_VOICE_ID:
        raise RuntimeError(
            "ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is not set"
        )
    for chunk in elevenlabs_client.text_to_speech.stream(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id=ELEVENLABS_MODEL_ID,
        output_format=ELEVENLABS_OUTPUT_FORMAT,
        optimize_streaming_latency=ELEVENLABS_LATENCY_MODE,
    ):
        if chunk:
            yield chunk


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json()
    user_message = data.get("message", "")
    print(f"You said: {user_message}")

    client = require_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0.8,
    )

    ai_reply = response.choices[0].message.content
    print(f"AI said: {ai_reply}")
    return jsonify({"reply": ai_reply})


@app.route("/realtime/session", methods=["POST"])
def realtime_session_route():
    require_openai_client()
    sdp_offer = request.get_data(as_text=True)
    if not sdp_offer:
        return jsonify({"error": "Missing SDP offer"}), 400

    files = {
        "sdp": (None, sdp_offer),
        "session": (None, json.dumps(build_realtime_session_config())),
    }

    response = requests.post(
        "https://api.openai.com/v1/realtime/calls",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        files=files,
        timeout=30,
    )

    if not response.ok:
        return (
            jsonify(
                {
                    "error": "Failed to start realtime session",
                    "details": response.text,
                }
            ),
            response.status_code,
        )

    flask_response = Response(response.text, mimetype="application/sdp")
    if "Location" in response.headers:
        flask_response.headers["X-OpenAI-Call-Location"] = response.headers["Location"]
    return flask_response


@app.route("/session", methods=["GET"])
def session_route():
    require_openai_client()
    response = requests.post(
        "https://api.openai.com/v1/realtime/client_secrets",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=build_realtime_client_secret_payload(),
        timeout=30,
    )

    return (
        jsonify(response.json()),
        response.status_code,
    )


@app.route("/speak", methods=["GET", "POST"])
def speak_route():
    text = get_speak_text()
    if not text:
        return jsonify({"error": "Missing text"}), 400
    if request.method == "GET" and is_duplicate_speak_request():
        return Response(status=204)

    def generate():
        provider = preferred_tts_provider()
        if provider == "elevenlabs":
            yield from stream_elevenlabs_audio(text)
            return
        yield from stream_openai_audio(text)

    return Response(
        stream_with_context(generate()),
        mimetype="audio/mpeg",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
        direct_passthrough=True,
    )


if __name__ == "__main__":
    print("✅ Server running at http://localhost:5000")
    app.run(debug=True, port=5000)
