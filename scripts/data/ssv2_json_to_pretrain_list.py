"""Convert an SSv2 json annotation file into a VideoMAE V2 pre-training list.

Pre-training ignores labels; each output line is one of
    <video_path> 0 -1                        (video files, decoded with decord)
    <frame_folder> <start_idx> <num_frames>  (extracted rawframes)

Usage (videos, e.g. 20bn-something-something-v2/78687.webm):
    python scripts/data/ssv2_json_to_pretrain_list.py \
        --json /gpfs/.../ssv2/labels/train.json \
        --video_dir /gpfs/.../ssv2/videos --ext webm \
        --out /gpfs/.../ssv2/pretrain_train.txt

Usage (rawframes, e.g. frames/78687/img_00001.jpg):
    python scripts/data/ssv2_json_to_pretrain_list.py \
        --json /gpfs/.../ssv2/labels/train.json \
        --frames_dir /gpfs/.../ssv2/frames \
        --out /gpfs/.../ssv2/pretrain_train.txt
"""
import argparse
import json
import os
import sys

ID_KEYS = ('id', 'video_id', 'vid', 'name', 'path', 'video')


def load_ids(json_path):
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # {"78687": {...}, ...} or {"database"/"videos": [...]}
        for key in ('videos', 'database', 'data', 'annotations'):
            if key in data:
                data = data[key]
                break
        else:
            return [str(k) for k in data.keys()], data
    if isinstance(data, dict):
        return [str(k) for k in data.keys()], data
    ids = []
    for entry in data:
        if isinstance(entry, (str, int)):
            ids.append(str(entry))
            continue
        for key in ID_KEYS:
            if key in entry:
                ids.append(str(entry[key]))
                break
        else:
            sys.exit('Could not find a video id in entry (looked for keys '
                     f'{ID_KEYS}): {json.dumps(entry)[:300]}')
    return ids, data


def strip_ext(vid):
    base = os.path.basename(vid)
    return os.path.splitext(base)[0] if '.' in base else base


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--json', required=True)
    p.add_argument('--out', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video_dir', help='folder with <id>.<ext> video files')
    src.add_argument('--frames_dir', help='folder with one <id>/ frame folder per clip')
    p.add_argument('--ext', default='webm', help='video extension (default webm)')
    p.add_argument('--start_idx', type=int, default=1,
                   help='index of the first frame file (default 1)')
    p.add_argument('--frame_exts', default='.jpg,.jpeg,.png')
    p.add_argument('--no_check', action='store_true',
                   help='do not check that files exist (videos only)')
    args = p.parse_args()

    ids, raw = load_ids(args.json)
    print(f'{len(ids)} entries in {args.json}; first entry: '
          f'{json.dumps(raw[0] if isinstance(raw, list) else ids[0])[:200]}')

    frame_exts = tuple(e.strip().lower() for e in args.frame_exts.split(','))
    lines, missing = [], []
    for vid in ids:
        name = strip_ext(vid)
        if args.video_dir:
            path = os.path.join(args.video_dir, f'{name}.{args.ext}')
            if not args.no_check and not os.path.isfile(path):
                missing.append(path)
                continue
            lines.append(f'{path} 0 -1')
        else:
            folder = os.path.join(args.frames_dir, name)
            if not os.path.isdir(folder):
                missing.append(folder)
                continue
            n = sum(1 for f in os.listdir(folder)
                    if f.lower().endswith(frame_exts))
            if n == 0:
                missing.append(folder)
                continue
            lines.append(f'{folder} {args.start_idx} {n}')

    if missing:
        print(f'WARNING: {len(missing)} clips missing/empty, skipped. '
              f'e.g. {missing[:3]}')
    if not lines:
        sys.exit('No clips found - check --video_dir/--frames_dir and --ext.')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {len(lines)} lines to {args.out}; first line: {lines[0]}')


if __name__ == '__main__':
    main()
