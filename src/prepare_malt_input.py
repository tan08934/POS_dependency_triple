"""
Prepare tokenized + POS-tagged sentences for MaltParser.

Supported input:
1) Sentence-level CSV with columns:
   - sent_id (optional), text (optional)
   - tokens: serialized list, e.g. "['Tôi' 'đọc' 'sách' '.']"
   - xpos or pos (serialized list)
2) Token-level CSV with columns:
   - sent_id, token, pos (one token per row)

Output is CoNLL-X/CoNLL-U-like 10-column input. HEAD/DEPREL are placeholders;
MaltParser should replace them during parsing.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path


_QUOTED = re.compile(r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"", re.S)


def parse_list_cell(value):
    """Parse a list stored in a CSV cell.

    Handles JSON lists, Python lists and numpy-style reprs such as
    ['Thời hạn' 'hưởng' '.'] (no commas, may span several lines).
    NOTE: ast.literal_eval must NOT be used for the numpy style, because
    adjacent string literals are silently concatenated into one string.
    """
    if value is None:
        return []
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return []

    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        # Works for both "['a', 'b']" and "['a' 'b']" (also empty strings).
        items = []
        for m in _QUOTED.finditer(value):
            raw = m.group(1) if m.group(1) is not None else m.group(2)
            items.append(raw.replace("\\'", "'").replace('\\"', '"'))
        if items or value.strip("[] \n") == "":
            return items

    # Last resort: whitespace-separated values
    return value.split()


def read_csv(path):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV không có header.")
        rows = list(reader)
        return reader.fieldnames, rows


def write_conll_sentence(out, sent_id, tokens, poses, text=""):
    if len(tokens) != len(poses):
        raise ValueError(
            f"{sent_id}: số token ({len(tokens)}) khác số POS ({len(poses)})."
        )
    for i, (token, pos) in enumerate(zip(tokens, poses), start=1):
        token = str(token).strip()
        pos = str(pos).strip() or "_"
        # CoNLL-X columns: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
        lemma = token.lower().replace(" ", "_")
        fields = [str(i), token.replace("\t", " "), lemma, pos, pos, "_", "0", "root", "_", "_"]
        out.write("\t".join(fields) + "\n")
    out.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV đầu ra từ Người 1 (đã tách từ + POS)")
    ap.add_argument("--output", default="data/malt_input.conll")
    ap.add_argument("--tokens-col", default="tokens")
    ap.add_argument("--pos-col", default="", help="Mặc định ưu tiên xpos, rồi pos, rồi upos")
    ap.add_argument("--id-col", default="sent_id")
    ap.add_argument("--text-col", default="text")
    args = ap.parse_args()

    fields, rows = read_csv(args.input)
    fieldset = set(fields)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Sentence-level rows (tokens and POS stored as arrays)
    if args.tokens_col in fieldset:
        pos_col = args.pos_col or next(
            (c for c in ("xpos", "pos", "upos") if c in fieldset), None
        )
        if not pos_col:
            raise ValueError("Không tìm thấy cột POS. Cần một trong: xpos, pos, upos.")
        with output.open("w", encoding="utf-8", newline="") as out:
            for index, row in enumerate(rows, start=1):
                tokens = parse_list_cell(row.get(args.tokens_col, ""))
                poses = parse_list_cell(row.get(pos_col, ""))
                if not tokens:
                    continue
                sid = row.get(args.id_col) or f"s{index:05d}"
                write_conll_sentence(out, sid, tokens, poses, row.get(args.text_col, ""))
    else:
        # Token-level rows
        token_col = "token" if "token" in fieldset else "form" if "form" in fieldset else None
        pos_col = args.pos_col or next((c for c in ("pos", "xpos", "upos") if c in fieldset), None)
        if not token_col or not pos_col or args.id_col not in fieldset:
            raise ValueError(
                "CSV cần cột tokens + POS dạng danh sách, hoặc dạng từng token với sent_id, token, pos."
            )
        grouped = {}
        for index, row in enumerate(rows, start=1):
            sid = row.get(args.id_col) or f"s{index:05d}"
            grouped.setdefault(sid, []).append((row.get(token_col, ""), row.get(pos_col, "_")))
        with output.open("w", encoding="utf-8", newline="") as out:
            for sid, pairs in grouped.items():
                write_conll_sentence(out, sid, [p[0] for p in pairs], [p[1] for p in pairs])

    print(f"Đã tạo: {output.resolve()}")
    print("Lưu ý: HEAD/DEPREL hiện là placeholder; chỉ dùng làm đầu vào cho bước MaltParser.")


if __name__ == "__main__":
    main()
