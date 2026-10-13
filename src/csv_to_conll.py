"""
Convert UDD-1.csv (treebank with gold head/deprel) into:
  - train.conll / test.conll : gold CoNLL-X files (train MaltParser / evaluate UAS-LAS)
  - input_pos.csv            : test sentences WITHOUT head/deprel (the only input the parser sees)

Usage:
  python src/csv_to_conll.py --input UDD-1.csv --outdir data --test-ratio 0.1
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_malt_input import parse_list_cell  # noqa: E402


def conll_block(tokens, upos, xpos, feats, heads, deprels):
    lines = []
    for i, (tok, up, xp, ft, h, d) in enumerate(zip(tokens, upos, xpos, feats, heads, deprels), start=1):
        tok = tok.strip().replace("\t", " ") or "_"
        lemma = tok.lower().replace(" ", "_")
        # CoNLL-X: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
        lines.append("\t".join([str(i), tok, lemma, up or "_", xp or "_", ft or "_", str(h), d or "_", "_", "_"]))
    return "\n".join(lines) + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="UDD-1.csv")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    good, skipped = [], 0
    for r in rows:
        cols = [parse_list_cell(r.get(c)) for c in ("tokens", "upos", "xpos", "feats", "head", "deprel")]
        n = len(cols[0])
        if n == 0 or any(len(c) != n for c in cols):
            skipped += 1
            continue
        try:
            heads = [int(h) for h in cols[4]]
        except ValueError:
            skipped += 1
            continue
        if sum(h == 0 for h in heads) != 1 or any(h < 0 or h > n for h in heads):
            skipped += 1
            continue
        good.append((r, cols[0], cols[1], cols[2], cols[3], heads, cols[5]))

    random.Random(args.seed).shuffle(good)
    n_test = max(1, int(len(good) * args.test_ratio))
    test, train = good[:n_test], good[n_test:]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.conll", train), ("test.conll", test)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as f:
            for _, *c in part:
                f.write(conll_block(*c))
    with (out / "input_pos.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sent_id", "text", "tokens", "xpos", "upos"])
        for r, toks, up, xp, *_ in part_rows(test):
            w.writerow([r["sent_id"], r["text"], repr(toks), repr(xp), repr(up)])

    print(f"Câu hợp lệ: {len(good)} | bỏ qua (lỗi định dạng): {skipped}")
    print(f"train={len(train)}  test={len(test)}  -> {out.resolve()}")


def part_rows(part):
    for r, toks, up, xp, ft, heads, deprels in part:
        yield r, toks, up, xp


if __name__ == "__main__":
    main()
