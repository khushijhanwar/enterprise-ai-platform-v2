"""
Measures a recording before it is sent for transcription.

"The transcript came back empty" has two very different causes: the
microphone delivered silence, or speech-to-text failed on real audio.
Measuring the recording's loudness tells them apart, and lets a silent
recording be rejected before it costs a transcription call.
"""
from __future__ import annotations

import io
import wave

import numpy as np

# Below this peak (fraction of full scale) a recording has no speech in
# it: a muted or virtual input device gives exact zeros, a live
# microphone's noise floor sits around 0.001.
SILENCE_PEAK = 0.005

_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}


def wav_stats(audio: bytes) -> dict | None:
    """Duration and peak level of a PCM WAV recording, or None when the
    bytes are not a WAV this can read (MP3, WebM, ...), in which case the
    caller should skip the check rather than guess."""
    try:
        with wave.open(io.BytesIO(audio)) as w:
            width, rate, frames = w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except (wave.Error, EOFError):
        return None
    if width not in _DTYPES or rate <= 0:
        return None

    samples = np.frombuffer(raw, dtype=_DTYPES[width]).astype(np.float64)
    if width == 1:                       # 8-bit WAV is unsigned, centred on 128
        samples -= 128.0
    full_scale = float(2 ** (8 * width - 1))
    peak = float(np.abs(samples).max() / full_scale) if samples.size else 0.0

    return {"duration_s": round(frames / rate, 2), "peak": round(peak, 4), "silent": peak < SILENCE_PEAK}
