import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from chat_app import DEFAULT_MODEL_PATH, make_handler

# Trained frozen-probe checkpoint written by train_frozen_probe.py (Stage 6).
# Tests below exercise the real winner; they skip if it hasn't been trained yet.
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "out_probe" / "best.pt"


class FakeRecommender:
    def __init__(self):
        self.context = None
        self.profile = None
        self.watch_history = None
        self.clarify = False
        self.confidence = 0.8

    def recommend(self, context, top_k=10):
        self.context = context
        return [
            {"movie_id": 123, "title": "Example Movie", "similarity": 0.75}
        ][:top_k]

    def recommend_with_context(
        self, context, top_k=10, profile="", watch_history=""
    ):
        self.context = context
        self.profile = profile
        self.watch_history = watch_history
        if self.clarify:
            return {
                "recommendations": [],
                "clarification": "Which genre do you prefer?",
                "confidence": 0.0,
                "preferences": [],
            }
        return {
            "recommendations": self.recommend(context, top_k),
            "clarification": None,
            "confidence": self.confidence,
            "preferences": ["mystery"],
        }


class ChatApiTests(unittest.TestCase):
    def test_chat_has_no_bundled_model_checkpoint(self):
        self.assertIsNone(DEFAULT_MODEL_PATH)

    @classmethod
    def setUpClass(cls):
        cls.recommender = FakeRecommender()
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(cls.recommender)
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, payload):
        connection = HTTPConnection("127.0.0.1", self.port)
        body = json.dumps(payload).encode("utf-8")
        connection.request(
            "POST",
            "/api/chat",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_chat_returns_movies_and_passes_conversation_context(self):
        status, result = self.post(
            {
                "message": "I like mysteries",
                "history": [{"speaker": "Seeker", "text": "I enjoyed a thriller"}],
                "profile": "Mysteries and thrillers",
                "watch_history": "Example Movie",
            }
        )

        self.assertEqual(status, 200)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertIn("Example Movie", result["reply"])
        self.assertEqual(
            self.recommender.context,
            [
                {"speaker": "Seeker", "text": "I enjoyed a thriller"},
                {"speaker": "Seeker", "text": "I like mysteries"},
            ],
        )
        self.assertEqual(self.recommender.profile, "Mysteries and thrillers")
        self.assertEqual(self.recommender.watch_history, "Example Movie")

    def test_clarification_can_be_answered_in_the_next_turn(self):
        self.recommender.clarify = True
        try:
            status, result = self.post({"message": "hello"})
        finally:
            self.recommender.clarify = False

        self.assertEqual(status, 200)
        self.assertEqual(result["reply"], "Which genre do you prefer?")

        status, result = self.post(
            {
                "message": "I like mysteries",
                "history": [
                    {"speaker": "Seeker", "text": "hello"},
                    {"speaker": "Recommender", "text": result["reply"]},
                ],
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertEqual(
            self.recommender.context[-2:],
            [
                {"speaker": "Recommender", "text": "Which genre do you prefer?"},
                {"speaker": "Seeker", "text": "I like mysteries"},
            ],
        )

    def test_chat_returns_clarification_instead_of_recommendations(self):
        self.recommender.clarify = True
        try:
            status, result = self.post({"message": "hello"})
        finally:
            self.recommender.clarify = False

        self.assertEqual(status, 200)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(result["reply"], "Which genre do you prefer?")

    def test_chat_rejects_empty_message(self):
        status, result = self.post({"message": "  "})

        self.assertEqual(status, 400)
        self.assertIn("error", result)


@unittest.skipUnless(
    CHECKPOINT_PATH.exists(),
    "trained checkpoint out_probe/best.pt not present (run train_frozen_probe.py)",
)
class RealCheckpointChatTests(unittest.TestCase):
    """Wire the Stage-6 winner (out_probe/best.pt) through make_handler."""

    @classmethod
    def setUpClass(cls):
        from chinese_recommender import ChineseRanker

        cls.ranker = ChineseRanker.load(CHECKPOINT_PATH)
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(cls.ranker)
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, payload=None):
        connection = HTTPConnection("127.0.0.1", self.port)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_health_endpoint_reports_ok(self):
        status, result = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(result, {"status": "ok"})

    def test_real_checkpoint_loads_full_catalog(self):
        self.assertEqual(len(self.ranker.order), 33834)
        self.assertEqual(len(self.ranker.movies), 33834)

    def test_real_checkpoint_serves_chinese_recommendations(self):
        status, result = self.request(
            "POST", "/api/chat", {"message": "我想看一部科幻电影"}
        )
        self.assertEqual(status, 200)
        self.assertIn("reply", result)
        self.assertGreater(len(result["recommendations"]), 0)
        top = result["recommendations"][0]
        self.assertTrue(top["title"])
        self.assertIsInstance(top["movie_id"], int)


if __name__ == "__main__":
    unittest.main()
