#!/usr/bin/env python3
"""Local browser chat shell for TG-ReDial. Bring your own recommender."""

from __future__ import annotations

import argparse
import json
import logging
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


PAGE_PATH = Path(__file__).resolve().parent / "static" / "chat.html"
MAX_REQUEST_BYTES = 64 * 1024
MAX_HISTORY_TURNS = 40
MAX_MESSAGE_LENGTH = 2000
# No bundled model. Bring your own recommender object exposing
# `recommend(context, top_k)` and optionally
# `recommend_with_context(context, top_k, profile, watch_history)`.
DEFAULT_MODEL_PATH: Path | None = None
logger = logging.getLogger("tgredial.chat")


def rerank_with_prompt(
    conversation: List[Dict[str, str]],
    candidates: List[Dict[str, Any]],
    profile: str = "",
    watch_history: str = "",
) -> List[Dict[str, Any]]:
    """Optionally rerank retrieved movies with an OpenAI-compatible chat API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not api_key or not model or not candidates:
        return candidates

    import urllib.error
    import urllib.request

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    prompt = f"""Choose the movies that best match the user's request and preferences.

Use only the candidates listed below. Do not invent movie titles or IDs.
Prioritize the user's latest request, then use earlier conversation and
watch history as supporting context. If nothing fits, return an empty list.

Conversation:
{json.dumps(conversation, ensure_ascii=False)}

User profile:
{profile or "Not provided"}

Watch history:
{watch_history or "Not provided"}

Candidates (the catalog provides titles only; no descriptions are available):
{json.dumps([{"id": item["movie_id"], "title": item["title"]} for item in candidates], ensure_ascii=False)}

Return JSON only:
{{"ranked_movie_ids": [123, 456, 789]}}"""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Follow the movie reranking instructions and return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        ranked_ids = json.loads(content).get("ranked_movie_ids", [])
        by_id = {int(item["movie_id"]): item for item in candidates}
        ordered = []
        seen = set()
        for movie_id in ranked_ids:
            try:
                movie_id = int(movie_id)
            except (TypeError, ValueError):
                continue
            if movie_id in by_id and movie_id not in seen:
                ordered.append(by_id[movie_id])
                seen.add(movie_id)
        return ordered
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError):
        logger.exception("LLM reranking failed; retaining ranker order")
        return candidates


def make_handler(recommender: Any) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                try:
                    body = PAGE_PATH.read_bytes()
                except OSError:
                    logger.exception("Could not read chat page at %s", PAGE_PATH)
                    self._send_json(500, {"error": "Chat page is unavailable."})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Not found."})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/chat":
                self._send_json(404, {"error": "Not found."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length header."})
                return
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "Request body is empty or too large."})
                return
            try:
                request = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid JSON."})
                return
            if not isinstance(request, dict):
                self._send_json(400, {"error": "Request body must be a JSON object."})
                return

            message = request.get("message")
            history = request.get("history", [])
            if not isinstance(message, str) or not message.strip():
                self._send_json(400, {"error": "Message must be a non-empty string."})
                return
            if len(message) > MAX_MESSAGE_LENGTH:
                self._send_json(
                    400,
                    {"error": f"Message must be at most {MAX_MESSAGE_LENGTH} characters."},
                )
                return
            if not isinstance(history, list) or len(history) > MAX_HISTORY_TURNS:
                self._send_json(
                    400,
                    {"error": f"History must be a list of at most {MAX_HISTORY_TURNS} turns."},
                )
                return

            context: List[Dict[str, str]] = []
            for turn in history:
                if not isinstance(turn, dict):
                    self._send_json(400, {"error": "Each history turn must be an object."})
                    return
                speaker = turn.get("speaker")
                text = turn.get("text")
                if speaker not in ("Seeker", "Recommender") or not isinstance(text, str):
                    self._send_json(
                        400,
                        {"error": "Each turn needs speaker Seeker/Recommender and text."},
                    )
                    return
                if len(text) > MAX_MESSAGE_LENGTH:
                    self._send_json(400, {"error": "A history turn is too long."})
                    return
                context.append({"speaker": speaker, "text": text})
            context.append({"speaker": "Seeker", "text": message.strip()})

            try:
                profile = request.get("profile", "")
                watch_history = request.get("watch_history", "")
                if not isinstance(profile, str) or not isinstance(watch_history, str):
                    self._send_json(400, {"error": "Profile and watch_history must be strings."})
                    return
                recommend_with_context = getattr(
                    recommender, "recommend_with_context", None
                )
                if callable(recommend_with_context):
                    decision = recommend_with_context(
                        context,
                        top_k=20,
                        profile=profile,
                        watch_history=watch_history,
                    )
                    recommendations = decision["recommendations"]
                    clarification = decision["clarification"]
                    confidence = decision["confidence"]
                    preferences = decision["preferences"]
                else:
                    recommendations = recommender.recommend(context, top_k=20)
                    clarification = None
                    confidence = None
                    preferences = []
                if not clarification:
                    recommendations = rerank_with_prompt(
                        context, recommendations, profile, watch_history
                    )[:5]
            except Exception:
                logger.exception("Recommendation request failed")
                self._send_json(500, {"error": "Could not generate recommendations."})
                return

            if clarification:
                reply = clarification
            elif recommendations:
                titles = ", ".join(str(item["title"]) for item in recommendations[:3])
                reply = f"Based on our conversation, you might enjoy: {titles}."
            else:
                reply = (
                    "I couldn't find a strong match from that message. "
                    "Tell me a genre, mood, or a movie you enjoyed, and I'll try again."
                )
            self._send_json(
                200,
                {
                    "reply": reply,
                    "recommendations": recommendations,
                    "confidence": confidence,
                    "preferences": preferences,
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    return ChatHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TG-ReDial browser chat.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser.error(
        "No recommender is bundled: import make_handler in your own server "
        "module and pass your model object in."
    )


if __name__ == "__main__":
    main()
