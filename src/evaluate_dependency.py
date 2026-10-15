#nguoi 2
import argparse
from pathlib import Path
import json


def read_conll(path):
    sentences, current = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 8 or "-" in cols[0] or "." in cols[0]:
                continue
            current.append(cols)
    if current:
        sentences.append(current)
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output", default="data/evaluation.json")
    args = ap.parse_args()

    gold = read_conll(args.gold)
    pred = read_conll(args.pred)
    if len(gold) != len(pred):
        raise ValueError(f"Số câu không khớp: gold={len(gold)}, pred={len(pred)}")

    total = correct_head = correct_labeled = 0
    for si, (gs, ps) in enumerate(zip(gold, pred), start=1):
        if len(gs) != len(ps):
            raise ValueError(f"Câu {si}: số token không khớp gold={len(gs)}, pred={len(ps)}")
        for gi, pi in zip(gs, ps):
            if gi[1] != pi[1]:
                raise ValueError(
                    f"Câu {si}: token không khớp: gold={gi[1]!r}, pred={pi[1]!r}"
                )
            total += 1
            same_head = gi[6] == pi[6]
            same_label = gi[7] == pi[7]
            correct_head += int(same_head)
            correct_labeled += int(same_head and same_label)

    result = {
        "sentences": len(gold),
        "tokens": total,
        "UAS": correct_head / total if total else 0.0,
        "LAS": correct_labeled / total if total else 0.0,
        "note": "Exclude punctuation only if the evaluation protocol explicitly requires it."
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
