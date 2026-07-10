"""
Setup diagnostic: checks which components are running at full strength
vs. on their offline fallback, and tells you exactly what to install to
close the gap. Run this any time -- it's read-only, makes no changes.

Run with: python check_setup.py
"""
import shutil
import subprocess
import sys


def check(label: str, ok: bool, fix: str = "") -> None:
    status = "OK  " if ok else "MISS"
    print(f"[{status}] {label}")
    if not ok and fix:
        print(f"        -> {fix}")


def main():
    print("Enterprise AI Platform -- setup check\n")

    # Java (required by PySpark)
    java_ok = shutil.which("java") is not None
    check("Java (required by PySpark)", java_ok, "Install a JRE, e.g. `apt install openjdk-21-jre-headless` (Linux) or `brew install openjdk` (Mac).")

    # PySpark import
    try:
        import pyspark  # noqa
        check(f"PySpark installed (v{pyspark.__version__})", True)
    except ImportError:
        check("PySpark installed", False, "pip install -r requirements.txt")

    # sentence-transformers + BGE model
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        try:
            SentenceTransformer("BAAI/bge-small-en-v1.5")
            check("BGE embedding model (BAAI/bge-small-en-v1.5)", True)
        except Exception as e:
            check(
                "BGE embedding model (BAAI/bge-small-en-v1.5)",
                False,
                f"Couldn't download model weights ({e}). Needs internet access to "
                f"huggingface.co on first run. Falling back to TF-IDF until then.",
            )
    except ImportError:
        check("sentence-transformers installed", False, "pip install -r requirements.txt")

    # Ollama server + model
    try:
        import ollama

        client = ollama.Client()
        models = client.list()
        model_names = [m.get("model", m.get("name", "")) for m in models.get("models", [])]
        has_qwen = any("qwen2.5" in m for m in model_names)
        check("Ollama server reachable", True)
        check(
            "qwen2.5:7b model pulled",
            has_qwen,
            "Run: ollama pull qwen2.5:7b",
        )
    except Exception:
        check(
            "Ollama server reachable",
            False,
            "Install from https://ollama.com, then run `ollama serve` (or just open the app) "
            "and `ollama pull qwen2.5:7b`. Falling back to extractive answers until then.",
        )

    print("\nEverything marked MISS has an automatic fallback -- the app will run either way.")
    print("This script just tells you what to install to get full-quality results.")


if __name__ == "__main__":
    main()
