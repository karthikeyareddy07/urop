import json
import argparse
import pickle
import re
import csv
from pathlib import Path
from typing import Any
from collections import Counter


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_csv_data(path):
    """Load a CSV file and return a list of dicts, parsing JSON string fields."""
    records = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse JSON string fields
            for key in ['messages', 'mentionMovies', 'goal_path']:
                if key in row and row[key]:
                    try:
                        row[key] = json.loads(row[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            records.append(row)
    return records


def save_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def normalize_text(value: Any) -> str:
    """Shared text normalization used by preprocessing.

    Lowercases, strips URLs and bracketed spans, and collapses whitespace
    so that preprocessing output is deterministic.
    """
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).lower()
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    return text.strip()


def tokenize(s):
    # simple tokenizer
    tokens = re.findall(r"\w+|[^\w\s]", s.lower(), flags=re.UNICODE)
    tokens = [t for t in tokens if not re.match(r'^\s*$', t)]
    return tokens


def lookup_value(rec, *keys):
    for key in keys:
        if key not in rec:
            continue
        value = rec[key]
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in ('none', 'nan', ''):
            continue
        return value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--out_dir', default='tgredial_processed')
    parser.add_argument('--keep_userids', action='store_true')
    parser.add_argument('--min_dialog_turns', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover files
    pkl_files = list(data_dir.glob('*.pkl'))
    csv_files = list(data_dir.glob('*.csv'))
    json_files = list(data_dir.glob('*.json'))

    found = { 'pkl': [str(p) for p in pkl_files], 'csv': [str(p) for p in csv_files], 'json': [str(p) for p in json_files] }
    print('Found files:', found)

    # load movies_with_mentions.csv if present
    movies_map = {}
    mfile = data_dir / 'movies_with_mentions.csv'
    if not mfile.exists():
        # try parent directory
        mfile = data_dir.parent / 'movies_with_mentions.csv'
    if mfile.exists():
        with open(mfile, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get('db_id') or row.get('movie_id') or row.get('id') or row.get('movieId')
                title = row.get('movieName') or row.get('title') or row.get('movie_title')
                if mid and title:
                    movies_map[str(mid)] = {'title': title}
    # load topic map if exists
    topic_map = {}
    tfile = data_dir / 'topic_to_id_2k5.json'
    if not tfile.exists():
        # try parent directory
        tfile = data_dir.parent / 'topic_to_id_2k5.json'
    if tfile.exists():
        with open(tfile, 'r', encoding='utf-8') as f:
            topic_map = json.load(f)

    # load user background (case-insensitive search)
    user2topic = {}
    user2_files = [p for p in pkl_files if 'user2topic' in p.name.lower()]
    user2_csv_files = [p for p in csv_files if 'user2topic' in p.name.lower() or 'usertopic' in p.name.lower()]
    if user2_files:
        try:
            user2topic = load_pickle(user2_files[0])
            print('Loaded user2topic from PKL:', user2_files[0])
        except Exception as e:
            print('Failed loading', user2_files[0], e)
    elif user2_csv_files:
        try:
            with open(user2_csv_files[0], 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row.get('user_id')
                    topics = row.get('topics')
                    if user_id and topics:
                        try:
                            topics_list = json.loads(topics)
                            user2topic[user_id] = topics_list
                        except (json.JSONDecodeError, TypeError):
                            pass
            print('Loaded user2topic from CSV:', user2_csv_files[0])
        except Exception as e:
            print('Failed loading', user2_csv_files[0], e)

    # Normalize user2topic keys to str (pkl uses str keys '0'; CSV branch
    # may store raw int keys; lookups use str(user_id)).
    try:
        user2topic = {str(k): v for k, v in user2topic.items()}
    except Exception:
        pass

    # Load per-split watch-history maps: {split}_identity2history.pkl with
    # keys 'dialog_id/mention_local_id' -> [movie global_ids].
    hist_maps: dict = {}
    for p in pkl_files:
        low = p.name.lower()
        if 'identity2history' not in low:
            continue
        split = 'train' if 'train' in low else ('valid' if ('valid' in low or 'dev' in low) else ('test' if 'test' in low else p.stem))
        try:
            hist_maps[split] = load_pickle(p)
            print(f'Loaded watch-history map for {split}: {p} ({len(hist_maps[split])} keys)')
        except Exception as e:
            print('Failed loading history map', p, e)

    # helper to process a split (PKL or CSV)
    def process_split(data_path, split_name):
        hist_map = hist_maps.get(split_name, {})
        if data_path.suffix == '.pkl':
            data = load_pickle(data_path)
        elif data_path.suffix == '.csv':
            data = load_csv_data(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        processed = []
        token_counts = Counter()
        dialogs = 0
        for rec_index, rec in enumerate(data):
            # try common keys
            raw_dialog_id = lookup_value(rec, 'dialog_id', 'dialogID', 'conv_id', 'convID', 'convId', 'conv', 'id')
            # normalize dialog id; if missing, create a stable fallback
            if raw_dialog_id is None or (isinstance(raw_dialog_id, str) and raw_dialog_id.strip().lower() in ('none','nan','')):
                dialog_id = f"{split}_conv_{rec_index}"
            else:
                dialog_id = str(raw_dialog_id)
            user_id = lookup_value(rec, 'user_id', 'userID', 'user')
            if user_id is None:
                user_id = 'unknown'

            # support TG-ReDial style: 'messages' list and 'mentionMovies' dict
            messages = rec.get('messages') or rec.get('dialog') or rec.get('utterances') or rec.get('turns') or []
            if isinstance(messages, dict):
                # unexpected dict -> skip
                messages = []

            # normalize message list: expect each item to have local_id, role, content
            norm_msgs = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                lid = m.get('local_id') or m.get('localId') or m.get('turn_id') or m.get('id')
                try:
                    lid = int(lid)
                except Exception:
                    # skip messages without numeric local id
                    continue
                role = m.get('role') or m.get('speaker') or ''
                text = m.get('content') or m.get('text') or m.get('utterance') or ''
                text = normalize_text(text)
                if len(text.strip())==0:
                    continue
                norm_msgs.append({'local_id': lid, 'role': role, 'text': text})

            if not norm_msgs:
                continue

            # sort messages by local_id
            norm_msgs = sorted(norm_msgs, key=lambda x: x['local_id'])

            mentionMovies = rec.get('mentionMovies') or rec.get('movie_mentions') or {}

            # helper to extract movie ids from mention value
            def extract_movie_ids(val):
                out = []
                if val is None:
                    return out
                if isinstance(val, str):
                    try:
                        out.append(int(val))
                    except:
                        pass
                    return out
                if isinstance(val, list):
                    # common TG-ReDial format: [global_id_str, title]
                    if len(val) >= 1 and isinstance(val[0], str) and val[0].isdigit():
                        try:
                            out.append(int(val[0]))
                        except:
                            pass
                        return out
                    # list of ids
                    for it in val:
                        try:
                            out.append(int(it))
                        except:
                            pass
                    return out
                if isinstance(val, dict):
                    # dict mapping to more info
                    for k, v in list(val.items()):
                        out.extend(extract_movie_ids(v))
                    return out
                return out

            # require at least some messages
            if len(norm_msgs) < args.min_dialog_turns:
                continue

            # for each mention turn, create a processed example where context is prior messages
            examples = []
            if isinstance(mentionMovies, dict):
                for lid_key, mv_val in mentionMovies.items():
                    try:
                        target_lid = int(lid_key)
                    except:
                        try:
                            target_lid = int(float(lid_key))
                        except:
                            continue
                    target_movie_ids = extract_movie_ids(mv_val)
                    if not target_movie_ids:
                        continue
                    # build context: messages with local_id < target_lid
                    context_msgs = [m for m in norm_msgs if m['local_id'] < target_lid]
                    if not context_msgs:
                        continue
                    # update token counts
                    for m in context_msgs:
                        toks = tokenize(m['text'])
                        for tk in toks:
                            token_counts[tk] += 1
                    examples.append({'dialog_id': str(dialog_id), 'user_id': str(user_id), 'context': [{'speaker': m['role'], 'text': m['text']} for m in context_msgs], 'target_movie_ids': target_movie_ids, 'mention_local_id': target_lid})

            if not examples:
                continue

            dialogs += 1
            # FIX (audit gap #1): join per-EXAMPLE so each mention turn gets
            # its own history (hist key = 'dialog_id/mention_local_id').
            # Profile lookup is str-normalized (user2topic keys are str).
            uinfo = user2topic.get(str(user_id))
            if uinfo is None:
                uinfo = user2topic.get(user_id)
            base_profile: list = []
            if isinstance(uinfo, dict):
                base_profile = list(uinfo.get('profile', uinfo.get('topics', [])))
            elif isinstance(uinfo, (list, tuple, set)):
                base_profile = list(uinfo)

            for ex in examples:
                ex_lid = ex.get('mention_local_id')
                ex_hist = hist_map.get(f"{dialog_id}/{ex_lid}", []) if hist_map else []
                ex['user_profile'] = base_profile
                ex['watch_history'] = list(ex_hist)
                processed.append(ex)
        # save
        out_path = out_dir / f'{split_name}.jsonl'
        save_jsonl(out_path, processed)
        return processed, token_counts

    # run on discovered data splits (PKL or CSV)
    data_splits = [p for p in pkl_files if '_data' in p.name.lower() or p.stem.lower() in ('train', 'valid', 'test')]
    data_splits.extend([p for p in csv_files if '_data' in p.name.lower() or p.stem.lower() in ('train', 'valid', 'test')])
    
    if not data_splits:
        # fallback: any file with list-of-dict structure will be considered
        data_splits = []
        for p in pkl_files:
            try:
                sample = load_pickle(p)
                if isinstance(sample, list):
                    data_splits.append(p)
            except Exception:
                continue
        for p in csv_files:
            try:
                sample = load_csv_data(p)
                if isinstance(sample, list):
                    data_splits.append(p)
            except Exception:
                continue

    files_created = []
    vocab_counter = Counter()
    stats = {}
    for p in data_splits:
        name = p.stem
        low = name.lower()
        if 'train' in low:
            split = 'train'
        elif 'valid' in low or 'dev' in low:
            split = 'valid'
        elif 'test' in low:
            split = 'test'
        else:
            split = name
        print('Processing', p, 'as', split)
        try:
            processed, tc = process_split(p, split)
        except Exception as e:
            print('Skipping', p, 'due to processing error:', e)
            continue
        files_created.append(str(out_dir / f'{split}.jsonl'))
        vocab_counter.update(tc)
        # processed is a list of per-mention examples; compute dialog-level stats
        if processed:
            unique_dialogs = set(p['dialog_id'] for p in processed)
            avg_context = sum(len(p.get('context', [])) for p in processed) / len(processed)
            stats[split] = {'dialogs': len(unique_dialogs), 'avg_context_messages': avg_context, 'examples': len(processed)}
        else:
            stats[split] = {'dialogs': 0, 'avg_context_messages': 0, 'examples': 0}

    # save users and movies
    users_out = out_dir / 'users.json'
    # convert non-serializable types (e.g., set) to lists
    safe_user2topic = {}
    for k, v in user2topic.items():
        if isinstance(v, (set, tuple)):
            safe_user2topic[k] = list(v)
        else:
            safe_user2topic[k] = v
    with open(users_out, 'w', encoding='utf-8') as f:
        json.dump(safe_user2topic, f, ensure_ascii=False, indent=2)
    files_created.append(str(users_out))
    movies_out = out_dir / 'movies.json'
    with open(movies_out, 'w', encoding='utf-8') as f:
        json.dump(movies_map, f, ensure_ascii=False, indent=2)
    files_created.append(str(movies_out))

    vocab_out = out_dir / 'vocab.json'
    with open(vocab_out, 'w', encoding='utf-8') as f:
        json.dump(vocab_counter.most_common(20000), f, ensure_ascii=False, indent=2)
    files_created.append(str(vocab_out))

    stats_out = out_dir / 'dataset_stats.json'
    with open(stats_out, 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'total_tokens': sum(vocab_counter.values())}, f, ensure_ascii=False, indent=2)
    files_created.append(str(stats_out))

    print('Created files:')
    for ff in files_created:
        print(' -', ff)

    print('\nProcessing complete. Sample output path:', out_dir)

if __name__=='__main__':
    main()
