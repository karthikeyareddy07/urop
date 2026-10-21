import pickle, json, csv, sys
from pathlib import Path

# Resolve the data directory in a portable way. Prefer a user-provided path,
# then the current workspace layout, and only fall back to a legacy absolute path.
preferred = None
if len(sys.argv) > 1:
    preferred = Path(sys.argv[1]).expanduser().resolve()
else:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / 'TG-ReDial-Data',
        Path.cwd() / 'TG-ReDial-Data',
        Path('c:/Users/Lenovo/Downloads/TG-ReDial-main/TG-ReDial-Data'),
    ]
    for c in candidates:
        if c.exists():
            preferred = c.resolve()
            break
if preferred is None:
    preferred = (Path(__file__).resolve().parents[1] / 'TG-ReDial-Data').resolve()

p = preferred
out = p / 'out'
out.mkdir(parents=True, exist_ok=True)

def lookup_value(rec, *keys):
    for key in keys:
        if key in rec:
            value = rec[key]
            if value is not None and not (isinstance(value, str) and value.strip().lower() in ('none', 'nan', '')):
                return value
    return None


def load_pkl(pth):
    with open(pth,'rb') as f:
        return pickle.load(f)

report = {}
# list files
report['files'] = [f.name for f in sorted(p.iterdir())]

# load raw splits
splits = {}
for name in ['train','valid','test']:
    fp = p/f'{name}_data.pkl'
    if fp.exists():
        data = load_pkl(fp)
        splits[name] = data
        report[f'{name}_convs'] = len(data)
    else:
        report[f'{name}_convs'] = None

# identity2history sizes
for name in ['train','valid','test']:
    fp = p/f'{name}_identity2history.pkl'
    if fp.exists():
        d=load_pkl(fp)
        report[f'{name}_identity2history_len']=len(d)
    else:
        report[f'{name}_identity2history_len']=None

# conv2user
conv2user=None
fp=p/'0619conv2user.pkl'
if fp.exists():
    conv2user=load_pkl(fp)
    report['conv2user_len']=len(conv2user)
else:
    report['conv2user_len']=0

# movies (read with csv to avoid pandas)
movies_fp=p/'movies_with_mentions.csv'
if movies_fp.exists():
    rows=0
    cols=None
    gids=[]
    with open(movies_fp,'r',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        cols=reader.fieldnames
        for r in reader:
            rows+=1
            if 'global_id' in r and r['global_id']!='':
                try:
                    gids.append(int(r['global_id']))
                except:
                    pass
    report['movies_rows']=rows
    if gids:
        report['global_id_min']=min(gids)
        report['global_id_max']=max(gids)
        report['global_id_unique']=len(set(gids))
    report['movies_columns']=cols
else:
    report['movies_rows']=0

# topic vocab
topic_fp=p/'topic_to_id_2k5.json'
if topic_fp.exists():
    with open(topic_fp,'r',encoding='utf-8') as f:
        topic=json.load(f)
    report['topic_count']=len(topic)
else:
    report['topic_count']=0

# user2topic coverage
u_fp=p/'user2TopicSent.pkl'
if u_fp.exists():
    u2=load_pkl(u_fp)
    report['user2topic_count']=len(u2)
else:
    report['user2topic_count']=0

# Check recommend-turns and identity2history mapping coverage
for name, data in splits.items():
    rec_turns=0
    mention_entries=0
    missing_keys=0
    conv_ids = set()
    for rec in data:
        # conv_id
        conv_id = lookup_value(rec, 'conv_id', 'convID', 'convId', 'conv', 'id')
        conv_ids.add(str(conv_id))
        mm = rec.get('mentionMovies',{})
        if isinstance(mm, dict):
            rec_turns += len(mm)
            mention_entries += sum(1 for k in mm)
            # check identity2history keys
            idmap = load_pkl(p/f'{name}_identity2history.pkl') if (p/f'{name}_identity2history.pkl').exists() else {}
            for lid in mm.keys():
                key=f"{conv_id}/{lid}"
                if key not in idmap:
                    missing_keys+=1
    report[f'{name}_rec_turns']=rec_turns
    report[f'{name}_mention_entries']=mention_entries
    report[f'{name}_missing_identity2history_keys']=missing_keys
    report[f'{name}_unique_conv_ids']=len(conv_ids)

# check processed outputs
for fn in ['train.jsonl','valid.jsonl','test.jsonl']:
    path = out/fn
    if path.exists():
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
        report[f'{fn}_lines']=len(lines)
        if len(lines)>0:
            try:
                j=json.loads(lines[0])
                report[f'{fn}_sample_keys']=list(j.keys())
            except:
                report[f'{fn}_sample_keys']='PARSE_ERROR'
    else:
        report[f'{fn}_lines']=0

# check for overlap of conv_ids across splits
train_ids = set(str(lookup_value(rec, 'conv_id', 'convID', 'convId', 'conv', 'id')) for rec in splits.get('train',[]))
valid_ids = set(str(lookup_value(rec, 'conv_id', 'convID', 'convId', 'conv', 'id')) for rec in splits.get('valid',[]))
test_ids  = set(str(lookup_value(rec, 'conv_id', 'convID', 'convId', 'conv', 'id')) for rec in splits.get('test',[]))
report['train_valid_overlap']=len(train_ids & valid_ids)
report['train_test_overlap']=len(train_ids & test_ids)
report['valid_test_overlap']=len(valid_ids & test_ids)

# save report
with open(out/'processing_check_report.json','w',encoding='utf-8') as f:
    json.dump(report,f,ensure_ascii=False,indent=2)

with open(out/'processing_check_report.md','w',encoding='utf-8') as f:
    f.write('# Processing Check Report\n\n')
    for k,v in report.items():
        f.write(f'- **{k}**: {v}\n')

print('Report written to', out)
