# TG-ReDial

We released a topic-guided conversational recommendation dataset for research on conversational/interactive recommender systems: TG-ReDial v1.0. It owns the following highlight:

## 1st -Multi-turn Topic-guided Conversation Recommendation

Our dataset incorporates topic threads to enforce natural semantic transitions towards the recommendation scenario. Starting with a non-recommendation topic, the topic thread naturally guides the user to the recommendation scenario through a sequence of evolving topics through multi-turn interaction.

## 2nd -Semi-automatic Annotated Dataset

We utilize ***real data*** to construct the recommended movies, topic threads, user profiles and utterances. And we propose semi-automatic approaches and novel rules to force human annotation into more reasonable and controllable.

## 3rd -Detailed User Background Information

For everybody wants to play our dataset, we are very glad to inform you that thanks to Yuanhang's hard work, ***our dataset, codes and learned parameters can be downloaded now!!!***

You can turn to the [Dataset Link](https://drive.google.com/drive/folders/1sWD6urkwyZo8ZyZBJoJw40eKK0jDNEni?usp=sharing) and the [Code and Parameters Link](https://github.com/RUCAIBox/TG_CRS_Code) for starting your travel in TG-ReDial!

### Conversational Recommendation Architecture

![TG–ReDial Conversational Recommendation Architecture](./Picture/TG-ReDial-Architecture.svg)

### Statistics

Overall dataset statistics (verified from raw `train_data.pkl`, `valid_data.pkl`, and `test_data.pkl`):

- Total conversations: 10,000
- Total dialogue turns/messages: 129,404
- Average conversation length: 12.94 turns
- Minimum conversation length: 10 turns
- Maximum conversation length: 16 turns

Split-wise statistics:

- Train: 8,495 conversations; 109,892 turns; average 12.94; min 10; max 16
- Validation: 757 conversations; 9,764 turns; average 12.90; min 10; max 16
- Test: 748 conversations; 9,748 turns; average 13.03; min 10; max 16

The original paper reports broader dataset-level summary numbers, but the raw released splits used in this repository consistently contain 10,000 conversations with 129,404 total dialogue turns/messages.

### Example

![Illustrative TG-ReDial conversation](./Picture/illustrative.png)

A major feature of our dataset is that we organize the conversation by topic threads, so that the transitions from chit-chat to recommendation are more natural. This is particularly useful for integrating our topics with open-domain conversations. Moreover, we associate a conversation with a unique user identity, so that it can closely resemble real-world cases. Especially, we can obtain profiles and watching history for the users in a conversation.

### Data Collection

![Project illustration](./Picture/DataCollection.png)

To our knowledge, most of existing datasets mainly focus on cold-start scenario for CRS, while it is also important that CRS can leverage historical interaction data for existing users. Our dataset provides the possibility of training conversational recommendation algorithms with historical interaction data. It is also feasible to study other personalized tasks, since a user is involved in multiple conversations in this dataset. Note that, in order to protect user privacy, we only sample users with a large number of watching records. For derived user

### Getting Started

This repository contains the TG-ReDial dataset tooling plus a browser chat
shell (`TG-ReDial-main/chat_app.py`, `TG-ReDial-main/static/chat.html`).
No recommender model is bundled: `chat_app.py` exposes
`make_handler(recommender)` and your own server module must pass in a
recommender object implementing `recommend(context, top_k)`.

#### Preprocess the data

```powershell
python TG-ReDial-main/preprocess_tgredial.py
```

Processed splits land in `TG-ReDial-Data/out/` (`train.jsonl`, `valid.jsonl`,
`test.jsonl`, `movies.json`, `users.json`).

#### Run tests

```powershell
python -m unittest test_chat_app -v
```

Run from `TG-ReDial-main/`.

### Model Status

A frozen `voidful/albert_chinese_tiny` encoder + linear-probe recommender has
been trained and shipped in this repository:

- Checkpoint: `out_probe/best.pt` (linear probe head; encoder frozen)
- Encoder: `voidful/albert_chinese_tiny` (loaded via HuggingFace at runtime)
- Catalog: 33,834 titles (`TG-ReDial-Data/out/`)
- Validation metrics: Hit@10 = 31.57%, NDCG@10 = 0.1536 (~2× random)
- Official test-split metrics: see `TG-ReDial-main/out_probe/test_metrics.json`

The model is wired into `serve_chinese.py` via
`chinese_recommender.ChineseRanker`, no API keys required.

#### Serve the model

```powershell
python serve_chinese.py --host 0.0.0.0 --port 8000 \
    --checkpoint out_probe/best.pt \
    --movies TG-ReDial-Data/out/movies.json
```

Endpoints:
- `GET /api/health` → `{"status": "ok"}`
- `GET /` → browser chat UI (`static/chat.html`)
- `POST /api/chat` → JSON body `{message, history, profile, watch_history}`,
  returns `{reply, recommendations, confidence, preferences}`

Open `http://localhost:8000/` in a browser for the end-to-end chat demo.

#### Run the official test-split evaluation

```powershell
python TG-ReDial-main/evaluate_test.py
```

Scores all 2,244 test examples against the full 33,834-title catalog and
writes `out_probe/test_metrics.json` (NDCG@10, Hit@10, Recall@20, MRR).

#### Run tests

```powershell
python -m unittest test_chat_app test_chinese_recommender -v
```

Run from `TG-ReDial-main/`. The `RealCheckpointChatTests` suite requires
`out_probe/best.pt` and the full catalog data under `TG-ReDial-Data/out/`.

#### Training CLI

`train_frozen_probe.py` accepts the following arguments (all optional with
sensible defaults):

```powershell
python TG-ReDial-main/train_frozen_probe.py --help
```

Key options: `--epochs`, `--lr`, `--batch`, `--data-dir`, `--train`, `--valid`,
`--out-dir`, `--seed`, `--patience`, `--max-neg`, `--ctx-cache`,
`--no-reuse-cache`, `--eval-test`.

Optional end-to-end fine-tuning: `--unfreeze-encoder` (unfreezes the ALBERT
encoder and trains it jointly with the probe head; contexts are encoded on the
fly) and `--encoder-lr` (learning rate for the encoder, default 1e-5). The
probe head uses `--lr`.

#### Dependency lock file

`requirements.txt` lists direct dependencies with pinned versions.
`requirements.lock.txt` is the full pinned environment (output of `pip freeze`)
and is the file to use for reproducible CI builds:

```powershell
pip install -r requirements.lock.txt
```

### Codes and Parameters Download

For the convenience of using this dataset for conversational recommendation task, we provide the code and learned parameters of all models in our paper "Towards Topic-Guided Conversational Recommender System", you can turn to the project [**TG-CRS-Code**](https://github.com/RUCAIBox/TG_CRS_Code) for acquiring details.

## Documentation

- `docs/model_card.md` — Model card: architecture, training hyperparameters,
  validation/test metrics, limitations, reproduction steps, ethical
  considerations, and references.

## Reference

If you use our dataset, please kindly cite our papers.
[Towards Topic-Guided Conversational Recommender System](https://arxiv.org/abs/2010.04125)

```c
@inproceedings{zhou2020topicguided,
  title={Towards Topic-Guided Conversational Recommender System},
  author={Kun Zhou and Yuanhang Zhou and Wayne Xin Zhao and Xiaoke Wang and Ji-Rong Wen},
  booktitle = {Proceedings of the 28th International Conference on Computational
               Linguistics, {COLING} 2020, Barcelona, Spain, December 8-11, 2020},
  year      = {2020}
}

```