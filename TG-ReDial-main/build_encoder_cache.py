"""Build frozen title-embedding cache for the Chinese encoder (Stage 4)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ENCODER = 'voidful/albert_chinese_tiny'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title-len', type=int, default=32)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'TG-ReDial-Data' / 'out'
    movies = json.load(open(out / 'movies.json', encoding='utf-8'))
    ids = sorted(movies.keys())
    npz = out / 'title_embeddings_albert_tiny.npz'
    order = out / 'movie_id_order.json'
    if npz.exists() and order.exists() and not a.force:
        print('cache exists, use --force to rebuild')
        return
    tok = AutoTokenizer.from_pretrained(ENCODER)
    m = AutoModel.from_pretrained(ENCODER)
    m.eval()
    thash = hashlib.sha1(tok.vocab.get('[UNK]', '') and str(sorted(tok.vocab.items())[:10]).encode() or b'x').hexdigest()[:12]
    print('tokenizer-hash', thash, flush=True)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(ids), a.batch):
            chunk = [movies[k].get('title', '') for k in ids[i:i + a.batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=a.title_len, return_tensors='pt')
            h = m(**enc).last_hidden_state
            mk = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
            v = (h * mk).sum(1) / mk.sum(1).clamp_min(1e-6)
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
            vecs.append(v.numpy().astype(np.float32))
            if (i // a.batch) % 20 == 0:
                print(f'{i}/{len(ids)}', flush=True)
    arr = np.concatenate(vecs, axis=0)
    np.savez_compressed(npz, embeddings=arr)
    json.dump({'ids': ids, 'encoder': ENCODER, 'tokenizer_hash': thash,
               'title_len': a.title_len, 'dim': arr.shape[1]}, open(order, 'w', encoding='utf-8'))
    print('saved', npz, arr.shape, flush=True)

if __name__ == '__main__':
    main()
