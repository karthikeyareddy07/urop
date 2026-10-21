# TG-ReDial

We released a topic-guided conversational recommendation dataset for research on conversational/interactive recommender systems: TG-ReDial v1.0. It owns the following highlight:

## 1st -Multi-turn Topic-guided Conversation Recommendation

Our dataset incorporates topic threads to enforce natural semantic transitions towards the recommendation scenario. Starting with a non-recommendation topic, the topic thread naturally guides the user to the recommendation scenario through a sequence of evolving topics through multi-turn interaction.

## 2nd -Semi-automatic Annotated Dataset

We utilize ***real data*** to construct the recommended movies, topic threads, user profiles and utterances. And we propose semi-automatic approaches and novel rules to force human annotation into more reasonable and controllable.

## 3rd -Detailed User Background Information

In our dataset, we incorporate each user with the user profile and interaction history, which are helpful to model the personality and preference of the user.

## Dataset Description

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

A major feature of our dataset is that we organize the conversation by topic threads, so that the transitions from chit-chat to recommendation are more natural. Such a dataset is particularly useful to help integrate the recommender component into general-purpose chat-bots, since it is easy to align our topics with open-domain conversations. Moreover, we associate a conversation with a unique user identity, so that it can closely resemble real-world cases. Especially, we can obtain profiles and watching history for the users in a conversation.

### Data Collection

![Project illustration](./Picture/DataCollection.png)

To our knowledge, most of existing datasets mainly focus on cold-start scenario for CRS, while it is also important that CRS can leverage historical interaction data for existing users. Our dataset provides the possibility of training conversational recommendation algorithms with historical interaction data. It is also feasible to study other personalized tasks, since a user is involved in multiple conversations in our dataset. Note that, in order to protect user privacy, we only sample users with a large number of watching records. For derived user data (e.g., profile or watching records), we perform the anonymized operation and add randomized modification (e.g., removal, replacement or deletion). We also require that the retrieved review sentences have to be written via paraphrasing. Finally, we ask human annotators to manually trace the user identities with corresponding user data in our dataset. We do not include the data from the users that can be identified in the final dataset.

## Dataset Download

You can get this dataset from [GoogleDrive](https://drive.google.com/drive/folders/1jLkNtUgzqBITQJsbOjSq20S2zzpY5Foj?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1fthFPz8Qjt54m4NR2G9AIA), the password for baidunetdisk is cc2o.

The dataset consists of four files:

## ?train_data.pkl, valid_data.pkl, test_data.pkl

the split train/valid/test data by the ratio of 8:1:1;

## ?User2TopicSent.pkl

the background information of each user, consisting of user id, interaction sequence and user profile.

## Preprocessing performed (by helper scripts)

I ran and debugged an end-to-end preprocessing pipeline to convert the raw TG-ReDial pickle files into model-ready JSONL and batching artifacts. Summary of actions and outputs:

- Code changes and scripts added (in `TG-ReDial-main`):
  - `preprocess_tgredial.py`: robust loader and converter. Handles the dataset schema (`messages`, `mentionMovies`), normalizes text, extracts per-mention training examples, and writes `out/train.jsonl`, `out/valid.jsonl`, `out/test.jsonl`, `out/users.json`, `out/movies.json`, `out/vocab.json`, and `out/dataset_stats.json`.
  - `check_pipeline.py`: validation script that scans raw files and processed outputs and writes `out/processing_check_report.json` and `out/processing_check_report.md`.
  - `show_samples.py`: prints 3 sample processed examples from each split.
  - `find_missing_keys.py`, `inspect_missing_detail.py`, `analyze_missing_keys_more.py`: utilities to locate and analyze missing identity->history mappings.
  - `batch_and_tokenize.py`: creates simple training batches under `out/batches/` and `out/batch_vocab.json` using a whitespace/token regex tokenizer.
- Key artifacts produced (location: `TG-ReDial-Data/out/`):
  - `train.jsonl` (25485 examples), `valid.jsonl` (2271), `test.jsonl` (2244)
  - `dataset_stats.json` — dialog/example counts and average context lengths
  - `vocab.json` — top tokens
  - `users.json`, `movies.json` — user-topic info and movie list
  - `processing_check_report.json` — automated verification (counts, missing keys, overlap checks)
  - `batches/` — 25 JSON batch files (each ~1024 examples) and `batch_vocab.json`
- Samples (3 per split, truncated):
  - train sample (example): context with seeker/recommender turns and `target_movie_ids` pointing to `1295242` ("精灵鼠小弟"), `mention_local_id` 7.
  - valid sample (example): `dialog_id`="10019", `user_id`="562", `target_movie_ids` includes `1302493`.
  - test sample (example): `dialog_id`="10034", `user_id`="563`,`target_movie_ids`includes`3529375`.
- Missing identity2history keys:
  - During an earlier run some processed examples had `dialog_id` equal to the string "None", which produced placeholder missing keys `None/7`, `None/11`, `None/15` in `out/missing_identity_keys.json`.
  - I analyzed the raw pickles and found the `train_identity2history.pkl` actually contains the required keys (e.g. `0/7`, `0/11`, `0/15`) — the `None` keys were artifacts of missing `conv_id` in the preprocessed output.
  - I patched `preprocess_tgredial.py` to generate stable fallback `dialog_id` values when `conv_id` is absent (pattern: `{split}_conv_{index}`), re-ran the preprocessing, and verified `processing_check_report.json` shows zero missing identity->history keys.
- Disk and environment notes:
  - The environment initially ran into "No space left on device" while installing heavy packages (`spacy`, `pandas`) and while writing large temp files. I freed large temp files under `%TEMP%` and re-ran preprocessing successfully. All scripts avoid `pandas`/`spacy` to keep the pipeline lightweight.

If you want, I can next:

- prepare more advanced tokenization (BPE / sentencepiece) and produce
  ready-to-train binary datasets, or
- run a quick local model training harness using the produced batches.

Files to inspect:

- `TG-ReDial-main/preprocess_tgredial.py` — main converter
- `TG-ReDial-Data/out/train.jsonl` — processed training examples
- `TG-ReDial-Data/out/processing_check_report.json` — verification report

Reply `bpe` or `none`.

## Data preprocessing and chat shell

Preprocess raw splits into `TG-ReDial-Data/out/` (`train.jsonl`,
`valid.jsonl`, `test.jsonl`, `movies.json`, `users.json`):

```powershell
.\.venv\Scripts\python.exe .\TG-ReDial-main\preprocess_tgredial.py
```

`TG-ReDial-main/chat_app.py` is a model-agnostic chat shell: import
`make_handler`, pass your own recommender object (exposing
`recommend(context, top_k)`), and serve it with your own server module.

## Codes and Parameters Download

For the convenience of using this dataset for conversational recommendation task, we provide the code and learned parameters of all models in our paper "Towards Topic-Guided Conversational Recommender System", you can turn to the project [**TG-CRS-Code**](https://github.com/RUCAIBox/TG_CRS_Code) for acquiring details.

## Reference

If you use our dataset, please kindly cite our papers.
[Towards Topic-Guided Conversational Recommender System](https://arxiv.org/abs/2010.04125)

```c
@inproceedings{zhou2020topicguided,
  title={Towards Topic-Guided Conversational Recommender System},
  author={Kun Zhou and Yuanhang Zhou and Wayne Xin Zhao and Xiaoke Wang and Ji-Rong Wen},
  booktitle = {Proceedings of the 28th International Conference on Computational
               Linguistics, {COLING} 2020, Barcelona, Spain, December 8-11,
               2020},
  year      = {2020}
}

```
