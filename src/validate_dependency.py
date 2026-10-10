#nguoi 2
import argparse
import csv
import json
import re
from pathlib import Path


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
            if len(cols) < 8:
                # Avoid silently misreading malformed rows
                raise ValueError(f"Dòng CoNLL không đủ 8 cột: {line}")
            # Skip range IDs and decimal IDs (UD multiword/empty nodes)
            if "-" in cols[0] or "." in cols[0]:
                continue
            current.append(cols)
    if current:
        sentences.append(current)
    return sentences


def sentence_type(text, n_tokens):
    lower = text.lower()
    labels = []
    if n_tokens <= 10:
        labels.append("short")
    else:
        labels.append("long")
    if re.search(r"\b(không|chưa|chẳng|chả|đừng|khỏi)\b", lower):
        labels.append("negative")
    if re.search(r"\b(bị|được)\b", lower):
        labels.append("passive_or_auxiliary")
    if len(re.findall(r"\b(?:là|có|đã|đang|sẽ|bị|được|khiến|làm|nói|cho biết)\b", lower)) >= 2:
        labels.append("multiple_verb_or_clause")
    return "|".join(labels) if labels else "other"


def get_text(rows):
    return " ".join(row[1] for row in rows if row[1] not in {".", ",", ";", ":", "!", "?"})


def inspect_sentence(rows):
    errors = []
    ids = set()
    for row in rows:
        try:
            ids.add(int(row[0]))
        except ValueError:
            errors.append(f"ID không phải số nguyên: {row[0]}")

    roots = []
    tokens = []
    for row in rows:
        try:
            tid = int(row[0])
            head = int(row[6])
        except ValueError:
            continue
        form, lemma, upos, xpos, feats, deprel = row[1], row[2], row[3], row[4], row[5], row[7]
        if head == 0:
            roots.append(tid)
        elif head not in ids:
            errors.append(f"head={head} không tồn tại cho token {tid} ({form})")
        if head == tid:
            errors.append(f"Token tự làm head: {tid} ({form})")
        if not deprel or deprel == "_":
            errors.append(f"Thiếu nhãn dependency cho token {tid} ({form})")
        tokens.append({
            "id": tid,
            "token": form,
            "lemma": lemma,
            "upos": upos,
            "pos": xpos,
            "head": head,
            "dependency": deprel,
        })

    if len(roots) != 1:
        errors.append(f"Số ROOT bất thường: {len(roots)} (ID={roots})")

    # Detect cycles by following head pointers.
    head_by_id = {}
    for row in rows:
        try:
            head_by_id[int(row[0])] = int(row[6])
        except ValueError:
            pass
    for start in head_by_id:
        seen = set()
        node = start
        while node in head_by_id and head_by_id[node] != 0:
            if node in seen:
                errors.append(f"Phát hiện chu trình dependency đi qua token ID {node}")
                break
            seen.add(node)
            node = head_by_id[node]

    return tokens, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File đầu ra MaltParser (.conll)")
    ap.add_argument("--jsonl", default="data/dependency_results.jsonl")
    ap.add_argument("--errors", default="data/dependency_errors.csv")
    args = ap.parse_args()

    input_path = Path(args.input)
    sentences = read_conll(input_path)
    json_path, error_path = Path(args.jsonl), Path(args.errors)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    error_rows = []
    with json_path.open("w", encoding="utf-8") as jf:
        for index, rows in enumerate(sentences, start=1):
            tokens, errors = inspect_sentence(rows)
            text = " ".join(t["token"] for t in tokens)
            sid = f"s{index:05d}"
            record = {
                "sent_id": sid,
                "sentence": text,
                "sentence_type": sentence_type(text, len(tokens)),
                "tokens": tokens,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            for error in errors:
                error_rows.append({
                    "sent_id": sid,
                    "sentence": text,
                    "sentence_type": record["sentence_type"],
                    "error_type": "structural",
                    "error_description": error,
                    "triple_impact": "needs_manual_review",
                    "corrected": "",
                })

    columns = [
        "sent_id", "sentence", "sentence_type", "error_type",
        "error_description", "triple_impact", "corrected"
    ]
    with error_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(error_rows)

    print(f"Số câu đã xử lý: {len(sentences)}")
    print(f"JSONL: {json_path.resolve()}")
    print(f"Structural errors: {len(error_rows)} -> {error_path.resolve()}")
    print("Lưu ý: kiểm tra cấu trúc không thay thế việc rà soát ngữ nghĩa thủ công.")


if __name__ == "__main__":
    main()
