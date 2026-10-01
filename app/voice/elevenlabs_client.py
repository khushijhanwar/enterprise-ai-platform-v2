"""
Voice I/O: ElevenLabs text-to-speech (answers out) and Scribe
speech-to-text (questions in).

Kept behind one small class for the same reason the LLM sits behind
`OllamaLLM`: the agents never import a vendor SDK directly, so the voice
provider is swappable and the platform still runs (text-only) when no
API key is configured.
"""
from __future__ import annotations

from app.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_STT_MODEL,
    ELEVENLABS_TTS_MODEL,
    ELEVENLABS_VOICE_ID,
)

AUDIO_MIME = "audio/mpeg"
_OUTPUT_FORMAT = "mp3_44100_128"


class VoiceClient:
    def __init__(
        self,
        api_key: str = ELEVENLABS_API_KEY,
        voice_id: str = ELEVENLABS_VOICE_ID,
        tts_model: str = ELEVENLABS_TTS_MODEL,
        stt_model: str = ELEVENLABS_STT_MODEL,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.tts_model = tts_model
        self.stt_model = stt_model
        self._client = None

    def is_available(self) -> bool:
        """True when an API key is configured. No network call, so a
        status check never spends credits."""
        return bool(self.api_key)

    def _sdk(self):
        if self._client is None:
            from elevenlabs.client import ElevenLabs

            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    def synthesize(self, text: str) -> bytes:
        """Text -> MP3 bytes."""
        chunks = self._sdk().text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.tts_model,
            output_format=_OUTPUT_FORMAT,
        )
        return b"".join(chunks)

    def transcribe(self, audio: bytes, filename: str = "question.wav", content_type: str = "audio/wav") -> str:
        """Recorded audio -> transcript text."""
        resp = self._sdk().speech_to_text.convert(
            model_id=self.stt_model,
            file=(filename, audio, content_type),
        )
        return (resp.text or "").strip()
