"""
Voice endpoints: the same LangGraph workflow as /api/chat, with speech
on both ends.

    POST /api/voice/ask        text question  -> spoken answer
    POST /api/voice/ask-audio  spoken question -> spoken answer
    GET  /api/voice/status     is voice configured?

Each response carries per-stage timings (transcribe / agent / speak) so
it is obvious where a slow voice turn is spending its time.
"""
from __future__ import annotations

import base64
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import VOICE_MAX_CHARS
from app.voice.audio_check import wav_stats
from app.voice.elevenlabs_client import AUDIO_MIME, VoiceClient
from app.voice.spoken import to_spoken_text

router = APIRouter()
_voice: VoiceClient | None = None


def get_voice() -> VoiceClient:
    global _voice
    if _voice is None:
        _voice = VoiceClient()
    return _voice


def get_agent():
    # Imported lazily: the agent pulls in the full RAG stack, and the
    # voice layer should not need it just to be imported or tested.
    from app.api.chat import get_agent as _get_agent

    return _get_agent()


class VoiceAskRequest(BaseModel):
    query: str


class VoiceResponse(BaseModel):
    query: str
    transcript: str | None = None      # set when the question arrived as audio
    tool_used: str
    spoken_text: str
    voice_enabled: bool
    audio_base64: str | None = None    # MP3; None when no API key is configured
    audio_mime: str = AUDIO_MIME
    timings_ms: dict[str, int]
    answer: str | None = None
    sql: str | None = None
    result: list[dict] | None = None
    sources: list[dict] | None = None


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _answer(query: str, transcript: str | None, timings: dict[str, int]) -> VoiceResponse:
    t = time.perf_counter()
    out = get_agent().handle(query)
    timings["agent"] = _ms(t)

    spoken = to_spoken_text(out, max_chars=VOICE_MAX_CHARS)

    voice = get_voice()
    audio_b64 = None
    if voice.is_available():
        t = time.perf_counter()
        try:
            audio_b64 = base64.b64encode(voice.synthesize(spoken)).decode("ascii")
        except Exception as e:  # quota, bad key, network
            raise HTTPException(status_code=502, detail=f"Text-to-speech failed: {str(e)[:300]}")
        timings["speak"] = _ms(t)

    is_sql = out["tool_used"] == "sql_query"
    return VoiceResponse(
        query=query,
        transcript=transcript,
        tool_used=out["tool_used"],
        spoken_text=spoken,
        voice_enabled=voice.is_available(),
        audio_base64=audio_b64,
        timings_ms=timings,
        sql=out.get("sql") if is_sql else None,
        result=out["result"].to_dict(orient="records") if is_sql else None,
        answer=None if is_sql else out.get("answer"),
        sources=None if is_sql else out.get("sources"),
    )


@router.get("/voice/status")
def voice_status() -> dict:
    voice = get_voice()
    return {
        "voice_enabled": voice.is_available(),
        "tts_model": voice.tts_model,
        "stt_model": voice.stt_model,
        "max_spoken_chars": VOICE_MAX_CHARS,
    }


@router.post("/voice/ask", response_model=VoiceResponse)
def voice_ask(req: VoiceAskRequest) -> VoiceResponse:
    """Text in, voice out. Without an API key this still returns the
    spoken-style text, with audio_base64 = None."""
    return _answer(req.query, transcript=None, timings={})


@router.post("/voice/ask-audio", response_model=VoiceResponse)
def voice_ask_audio(file: UploadFile = File(...)) -> VoiceResponse:
    """Voice in, voice out."""
    voice = get_voice()
    if not voice.is_available():
        raise HTTPException(status_code=503, detail="Voice is not configured: set ELEVENLABS_API_KEY in .env")

    audio = file.file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    # Measure before transcribing: a silent recording is a microphone
    # problem, not a speech problem, and should not cost a transcription.
    stats = wav_stats(audio)
    if stats and stats["silent"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The recording is silent: {stats['duration_s']} s long, peak level {stats['peak']:.2%}. "
                "The browser recorded from an input that delivered no sound, so nothing was sent for "
                "transcription. Check which microphone the browser is using."
            ),
        )

    timings: dict[str, int] = {}
    t = time.perf_counter()
    try:
        transcript = voice.transcribe(audio, file.filename or "question.wav", file.content_type or "audio/wav")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Speech-to-text failed: {str(e)[:300]}")
    timings["transcribe"] = _ms(t)

    if not transcript:
        heard = f" ({stats['duration_s']} s, peak level {stats['peak']:.0%})" if stats else ""
        raise HTTPException(
            status_code=422,
            detail=f"The recording has sound{heard}, but speech-to-text returned no words for it",
        )

    return _answer(transcript, transcript=transcript, timings=timings)
