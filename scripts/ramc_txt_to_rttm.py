"""MagicData-RAMC ships its own timestamped speaker annotation as TXT, not
RTTM (DiariZen's recipe does not ship RAMC labels). Convert it so the same
prep_diarizen_dataset.py path can consume it.

RAMC TXT line format (per session, see its README.txt):
    [start_time,end_time]\tspeaker_id\tgender,language\ttranscription
`[+]` inside the transcription marks overlapping speech; the per-utterance
timestamps already encode the overlap, so no special handling is needed —
frame counts come from the per-speaker union.

Example:
  python scripts/ramc_txt_to_rttm.py \
      --root "/media/.../ramc/extracted" \
      --split-tsv "/media/.../ramc/extracted/DataPartition/train.tsv" \
      --out-rttm "/media/.../ramc/rttm/train.rttm"
"""
import argparse
import glob
import os
import re

LINE_RE = re.compile(r"^\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding MDT2021S* session dirs")
    ap.add_argument("--split-tsv", default=None,
                    help="DataPartition tsv listing the wav basenames of a split")
    ap.add_argument("--out-rttm", required=True)
    args = ap.parse_args()

    keep = None
    if args.split_tsv:
        keep = set()
        for line in open(args.split_tsv):
            parts = line.split()
            if parts and parts[0].endswith(".wav"):
                keep.add(os.path.splitext(os.path.basename(parts[0]))[0])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_rttm)), exist_ok=True)
    n_sess = n_seg = n_skip = 0
    with open(args.out_rttm, "w") as out:
        for txt in sorted(glob.glob(os.path.join(args.root, "MDT*", "TXT", "*.txt"))):
            rid = os.path.splitext(os.path.basename(txt))[0]
            if keep is not None and rid not in keep:
                n_skip += 1
                continue
            wrote = False
            with open(txt, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = LINE_RE.match(line)
                    if not m:
                        continue
                    s, e, spk = float(m.group(1)), float(m.group(2)), m.group(3)
                    if e <= s:
                        continue
                    out.write(f"SPEAKER {rid} 1 {s:.3f} {e - s:.3f} "
                              f"<NA> <NA> {spk} <NA> <NA>\n")
                    n_seg += 1
                    wrote = True
            n_sess += wrote
    print(f"RAMC -> {args.out_rttm}: {n_sess} sessions, {n_seg} segments "
          f"({n_skip} sessions not in split)")


if __name__ == "__main__":
    main()
