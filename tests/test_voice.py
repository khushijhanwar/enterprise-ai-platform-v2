"""
Tests for the voice layer. No network, no API key, no models: the
ElevenLabs client and the agent are replaced with fakes, so these run
in CI in under a second.

    pytest tests/test_voice.py
"""
import base64

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import voice as voice_api
from app.voice.spoken import to_spoken_text, truncate_at_sentence

SQL_OUTPUT = {
    "tool_used": "sql_query",
    "sql": "SELECT account_name, monthly_spend_usd FROM accounts ORDER BY 2 DESC LIMIT 2",
    "result": pd.DataFrame(
        [
            {"account_id": "ACC-1", "account_name": "Customer 12", "monthly_spend_usd": 19982.24},
            {"account_id": "ACC-2", "account_name": "Customer 7", "monthly_spend_usd": 18551.68},
        ]
    ),
}

DOC_OUTPUT = {
    "tool_used": "document_retrieval",
    "answer": "Refunds are issued within **14 days** of purchase [refund_policy.txt]. Contact support to start one.",
    "sources": [{"source": "data/sample_docs/refund_policy.txt", "score": 0.91, "text": "..."}],
}


# ---------- spoken text ----------

def test_sql_rows_become_a_sentence_not_a_table():
    text = to_spoken_text(SQL_OUTPUT)
    assert "Customer 12 with 19,982 dollars" in text
    assert "Customer 7 with 18,552 dollars" in text
    assert "account_id" not in text and "{" not in text


def test_single_value_result():
    out = {"tool_used": "sql_query", "result": pd.DataFrame([{"account_count": 42}])}
    assert to_spoken_text(out) == "The account count is 42."


def test_empty_sql_result():
    out = {"tool_used": "sql_query", "result": pd.DataFrame()}
    assert "no rows" in to_spoken_text(out)


def test_long_results_only_speak_the_first_five():
    rows = [{"account_name": f"Customer {i}", "api_calls_last_30d": 1000 - i} for i in range(8)]
    text = to_spoken_text({"tool_used": "sql_query", "result": rows}, max_chars=1000)
    assert "Customer 4" in text and "Customer 5" not in text
    assert "The other 3 are on screen." in text


def test_document_answer_drops_markdown_and_names_its_source():
    text = to_spoken_text(DOC_OUTPUT)
    assert "*" not in text and "[" not in text
    assert text.startswith("Refunds are issued within 14 days of purchase. Contact")
    assert text.endswith("Source: refund policy.")


def test_citation_survives_truncation():
    out = dict(DOC_OUTPUT, answer="This is a sentence. " * 100)
    text = to_spoken_text(out, max_chars=120)
    assert len(text) <= 120
    assert text.endswith("Source: refund policy.")


def test_truncation_ends_on_a_full_sentence():
    text = truncate_at_sentence("First sentence here. Second sentence is much longer than the rest.", 40)
    assert text == "First sentence here."


# ---------- endpoints ----------

class FakeAgent:
    def __init__(self, output):
        self.output, self.queries = output, []

    def handle(self, query):
        self.queries.append(query)
        return self.output


class FakeVoice:
    tts_model, stt_model = "fake-tts", "fake-stt"

    def __init__(self, available=True, heard="How do refunds work?"):
        self.available, self.heard, self.spoken = available, heard, []

    def is_available(self):
        return self.available

    def synthesize(self, text):
        self.spoken.append(text)
        return b"MP3" + text.encode()

    def transcribe(self, audio, filename="q.wav", content_type="audio/wav"):
        return self.heard


@pytest.fixture
def client(monkeypatch):
    def make(agent_output=DOC_OUTPUT, voice=None):
        agent, voice = FakeAgent(agent_output), voice or FakeVoice()
        monkeypatch.setattr(voice_api, "get_agent", lambda: agent)
        monkeypatch.setattr(voice_api, "get_voice", lambda: voice)
        app = FastAPI()
        app.include_router(voice_api.router, prefix="/api")
        return TestClient(app), agent, voice

    return make


def test_text_question_gets_a_spoken_answer(client):
    http, agent, voice = client()
    body = http.post("/api/voice/ask", json={"query": "How do refunds work?"}).json()

    assert body["tool_used"] == "document_retrieval"
    assert base64.b64decode(body["audio_base64"]).startswith(b"MP3")
    assert voice.spoken == [body["spoken_text"]]       # what was billed is what is shown
    assert "speak" in body["timings_ms"] and "agent" in body["timings_ms"]


def test_spoken_question_is_transcribed_then_routed(client):
    http, agent, voice = client(agent_output=SQL_OUTPUT, voice=FakeVoice(heard="Top accounts by spend?"))
    r = http.post("/api/voice/ask-audio", files={"file": ("q.wav", b"RIFFfake", "audio/wav")})
    body = r.json()

    assert r.status_code == 200
    assert body["transcript"] == "Top accounts by spend?"
    assert agent.queries == ["Top accounts by spend?"]  # the agent got the transcript
    assert body["tool_used"] == "sql_query" and len(body["result"]) == 2
    assert set(body["timings_ms"]) == {"transcribe", "agent", "speak"}


def test_no_api_key_falls_back_to_text_only(client):
    http, _, voice = client(voice=FakeVoice(available=False))
    body = http.post("/api/voice/ask", json={"query": "How do refunds work?"}).json()

    assert body["voice_enabled"] is False and body["audio_base64"] is None
    assert body["spoken_text"]                           # still answers
    assert voice.spoken == []                            # and spends nothing

    r = http.post("/api/voice/ask-audio", files={"file": ("q.wav", b"x", "audio/wav")})
    assert r.status_code == 503


def test_silent_recording_is_rejected_before_any_speech_is_billed(client):
    http, agent, voice = client(voice=FakeVoice(heard=""))
    r = http.post("/api/voice/ask-audio", files={"file": ("q.wav", b"x", "audio/wav")})
    assert r.status_code == 422
    assert agent.queries == [] and voice.spoken == []


def test_provider_failure_is_a_clean_502(client):
    class Broken(FakeVoice):
        def synthesize(self, text):
            raise RuntimeError("quota_exceeded")

    http, _, _ = client(voice=Broken())
    r = http.post("/api/voice/ask", json={"query": "How do refunds work?"})
    assert r.status_code == 502 and "quota_exceeded" in r.json()["detail"]
