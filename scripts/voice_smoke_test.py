"""
Round-trip check for the voice layer, independent of the rest of the app:

    text --(ElevenLabs text-to-speech)--> answer.mp3 --(Scribe)--> text

Run from the repo root:  python scripts/voice_smoke_test.py
Costs well under 100 credits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.voice.elevenlabs_client import VoiceClient  # noqa: E402

SENTENCE = "The top account by spend is Customer 12."

voice = VoiceClient()
if not voice.is_available():
    sys.exit("ELEVENLABS_API_KEY is not set. Add it to .env first.")

audio = voice.synthesize(SENTENCE)
Path("answer.mp3").write_bytes(audio)
print(f"spoke:  {SENTENCE}")
print(f"saved:  answer.mp3 ({len(audio):,} bytes)")

heard = voice.transcribe(audio, "answer.mp3", "audio/mpeg")
print(f"heard:  {heard}")
