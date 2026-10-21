import json
import unittest
from pathlib import Path

import numpy as np
import torch

from chinese_recommender import (
    ChineseRanker, ChineseRecommender, build_context_text, ENCODER_NAME, VERSION,
)


class ModelTests(unittest.TestCase):
    def test_context_text_uses_last_turns(self):
        ctx = [{'speaker': 'Seeker', 'text': f'msg{i}'} for i in range(12)]
        s = build_context_text(ctx)
        self.assertIn('msg11', s)
        self.assertNotIn('msg0', s)

    def test_forward_shape(self):
        m = ChineseRecommender(enc_dim=312, movie_count=50)
        c = torch.randn(4, 312)
        t = torch.randn(50, 312)
        out = m(c, t)
        self.assertEqual(list(out.shape), [4, 50])

    def test_determinism(self):
        torch.manual_seed(0)
        m = ChineseRecommender(enc_dim=312, movie_count=10)
        m.eval()
        c = torch.randn(2, 312)
        t = torch.randn(10, 312)
        with torch.no_grad():
            a = m(c, t)
            b = m(c, t)
        self.assertTrue(torch.allclose(a, b))

    def test_recommend_topk_shape(self):
        movies = {str(i): {'title': f'电影{i}'} for i in range(20)}
        emb = np.random.randn(20, 312).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        order = [str(i) for i in range(20)]

        class FakeTok:
            def __call__(self, texts, **kw):
                import torch as T
                n = len(texts)
                return {'input_ids': T.zeros(n, 4, dtype=T.long),
                        'attention_mask': T.ones(n, 4)}

        class FakeEnc:
            def __call__(self, **kw):
                import torch as T
                n = kw['input_ids'].shape[0]
                class O:
                    last_hidden_state = T.randn(n, 4, 312)
                return O()

        m = ChineseRecommender(enc_dim=312, movie_count=20)
        m.eval()
        r = ChineseRanker(m, movies, emb, order, FakeTok(), FakeEnc())
        recs = r.recommend([{'speaker': 'Seeker', 'text': '科幻'}], top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertIn('movie_id', recs[0])
        d = r.recommend_with_context([{'speaker': 'Seeker', 'text': '科幻'}])
        self.assertIn('recommendations', d)
        self.assertIn('clarification', d)


if __name__ == '__main__':
    unittest.main()
