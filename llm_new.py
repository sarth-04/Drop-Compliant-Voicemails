import os
import wave
import json
import threading
import time
import numpy as np
from dotenv import load_dotenv

# ================= ENV =================
load_dotenv(override=True)

# ================= DEPENDENCIES =================
try:
    from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
except ImportError:
    print("⚠️  Deepgram SDK missing. Run: pip install deepgram-sdk==3.*")

# --- Gemini ---
HAS_GEMINI = False
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass

# --- OpenAI ---
HAS_OPENAI = False
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    pass

# ================= CONFIG =================
CONFIG = {
    "CHUNK_MS": 50,
    "SILENCE_RMS_THRESH": 100,

    "DEFAULT_WAIT_TIME": 2.2,
    "INITIAL_WAIT_TIME": 4.0,
    "LLM_CONFIRMED_WAIT_TIME": 0.6,

    "GEMINI_MODEL": "gemini-2.5-flash",
    "OPENAI_MODEL": "gpt-4o-mini",
}

# ================= LLM ANALYZER =================
class LLMAnalyzer:
    def __init__(self):
        self.client = None
        self.provider = None
        self.history = ""
        self.is_greeting_finished = False

        # ---- PRIORITY 1: GEMINI ----
        if HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(CONFIG["GEMINI_MODEL"])
            self.provider = "gemini"
            print("   🧠 AI Initialized: Gemini")

        # ---- PRIORITY 2: OPENAI ----
        elif HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.provider = "openai"
            print("   🧠 AI Initialized: OpenAI")

        else:
            print("   ⚠️ No Gemini/OpenAI API key found. AI disabled.")

    def analyze_text(self, new_text):
        if self.is_greeting_finished:
            return

        self.history += " " + new_text
        if not self.client:
            return

        prompt = (
            "Analyze this voicemail greeting. "
            "Return JSON {\"finished\": boolean}. "
            "Set finished=true ONLY if the speaker has finished speaking "
            "and is clearly asking the caller to leave a message.\n\n"
            f"Transcript: {self.history}"
        )

        try:
            if self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                result = json.loads(response.text)

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=CONFIG["OPENAI_MODEL"],
                    messages=[
                        {"role": "system", "content": "You analyze voicemail greetings."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                result = json.loads(response.choices[0].message.content)

            else:
                return

            if result.get("finished"):
                print("   🧠 [AI] Greeting Finished Detected")
                self.is_greeting_finished = True

        except Exception as e:
            print(f"   ⚠️ AI Error ({self.provider}): {e}")

# ================= DECISION ENGINE =================
class DecisionEngine:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.llm_analyzer = LLMAnalyzer()

        self.last_speech_time = 0
        self.has_speech_started = False

    def process_frame(self, mono_data, timestamp):
        rms = np.sqrt(np.mean(mono_data.astype(float) ** 2))
        is_silence = rms < CONFIG["SILENCE_RMS_THRESH"]

        if not is_silence:
            self.last_speech_time = timestamp
            if not self.has_speech_started:
                print(f"   🗣️ Audio detected at {timestamp:.2f}s")
                self.has_speech_started = True
        else:
            silence_duration = timestamp - self.last_speech_time

            if not self.has_speech_started:
                limit = CONFIG["INITIAL_WAIT_TIME"]
                reason = "Initial Silence"
            elif self.llm_analyzer.is_greeting_finished:
                limit = CONFIG["LLM_CONFIRMED_WAIT_TIME"]
                reason = "AI Confirmed End"
            else:
                limit = CONFIG["DEFAULT_WAIT_TIME"]
                reason = "Standard Silence"

            if silence_duration >= limit and timestamp > 1.0:
                print(f"   ✅ DROP REASON: {reason} ({silence_duration:.1f}s)")
                return timestamp

        return None

    def on_transcript(self, text):
        threading.Thread(
            target=self.llm_analyzer.analyze_text,
            args=(text,),
            daemon=True
        ).start()

# ================= MAIN SIMULATION =================
def run_simulation(file_path):
    print(f"\n📞 Dialing {file_path}...")

    if not os.path.exists(file_path):
        print("❌ Audio file not found.")
        return

    wf = wave.open(file_path, "rb")
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()

    print(f"   Audio: {sample_rate}Hz | {channels} channel(s)")

    engine = DecisionEngine(sample_rate)

    # ---- Deepgram ----
    dg_socket = None
    if os.getenv("DEEPGRAM_API_KEY"):
        try:
            dg_client = DeepgramClient()
            dg_socket = dg_client.listen.websocket.v("1")

            def on_message(self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if sentence:
                    print(f"   🗣️ Transcript: {sentence}")
                    engine.on_transcript(sentence)

            dg_socket.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2-phonecall",
                encoding="linear16",
                sample_rate=sample_rate,
                channels=channels,
            )

            dg_socket.start(options)
        except Exception as e:
            print(f"   ⚠️ Deepgram Error: {e}")

    # ---- Streaming Loop ----
    chunk_frames = int(sample_rate * (CONFIG["CHUNK_MS"] / 1000))
    current_time = 0.0

    try:
        while True:
            raw_bytes = wf.readframes(chunk_frames)
            if not raw_bytes:
                break

            if dg_socket:
                dg_socket.send(raw_bytes)

            data = np.frombuffer(raw_bytes, dtype=np.int16)
            if channels == 2:
                data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

            current_time += len(data) / sample_rate

            drop_time = engine.process_frame(data, current_time)
            if drop_time:
                print(f"🚀 ACTION: DROP MESSAGE at {drop_time:.2f}s")
                break

            time.sleep(CONFIG["CHUNK_MS"] / 1000)

    finally:
        if dg_socket:
            dg_socket.finish()
        wf.close()

# ================= ENTRY =================
if __name__ == "__main__":
    run_simulation("dataset/vm1_output.wav")
