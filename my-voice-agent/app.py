from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from groq import Groq
from elevenlabs.client import ElevenLabs
import datetime
import pytz
import io

app = Flask(__name__)
CORS(app)

# ── Groq (AI Brain) ──
groq_client = Groq(
    api_key="" 
)

# ── ElevenLabs (Human Voice) ──
eleven_client = ElevenLabs(
    api_key=""
)

SYSTEM_PROMPT = """
You are a helpful friendly AI assistant.
Answer questions naturally and conversationally.
Only mention time or date if the user specifically asks.
Keep responses short — max 2 sentences.
Always respond in English.
"""

@app.route("/")
def index():
    return send_file("index.html")

# ── Route 1: Get AI text response ──
@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json()
    user_message = data.get("message", "")

    print(f"You said: {user_message}")

    india_time = datetime.datetime.now(
        pytz.timezone("Asia/Kolkata")
    ).strftime("%I:%M %p, %A %d %B %Y")

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + f"\nCurrent India time: {india_time}"
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    ai_reply = response.choices[0].message.content
    print(f"AI said: {ai_reply}")

    return jsonify({"reply": ai_reply})

# ── Route 2: Convert text to human voice ──
@app.route("/speak", methods=["POST"])
def speak_route():
    data = request.get_json()
    text = data.get("text", "")

    print(f"Speaking: {text}")

    # ElevenLabs TTS
    audio = eleven_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George voice
        text=text,
        model_id="eleven_flash_v2_5",      # Fastest free model
        output_format="mp3_44100_128",
    )

    # Convert generator to bytes
    audio_bytes = b"".join(audio)
    return Response(audio_bytes, mimetype="audio/mpeg")

if __name__ == "__main__":
    print("✅ Server running at http://localhost:5000")
    app.run(debug=True, port=5000)