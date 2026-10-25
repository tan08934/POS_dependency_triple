#nguoi 2
import argparse
import ast
import csv
import json
import re
from pathlib import Path
import joblib

from csv_to_conll import parse_list_cell
from pos_tagging import features, sentfeatures, predict
from validate_dependency import read_conll



def read_csv(path):
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
    ap.add_argument("--input", default=None, help="CSV có token và POS nếu dùng chế độ CSV cũ.")
    ap.add_argument("--test-gold", default="data/test.conll", help="Lấy token và ranh giới câu từ tập test.")
    ap.add_argument("--output", default="data/malt_input.conll")
    args = ap.parse_args()


   # Lấy đường dẫn từ thư mục gốc của dự án.
    root_dir = Path(__file__).resolve().parent.parent

    test_path = root_dir / args.test_gold
    output_path = root_dir / args.output

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tập test: {test_path}"
        )

    # Chỉ đọc token và UPOS Gold từ tập test; file này không dùng CRF.
    sentences = read_conll(test_path)

    if not sentences:
        raise ValueError("Tập test không có câu hợp lệ.")

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    total_tokens = 0

    with output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:

        for index, sentence_rows in enumerate(
            sentences, start=1
        ):
            # Cột 2 của CoNLL-X là FORM (token).
            tokens = [row[1] for row in sentence_rows]

            # CoNLL-X: cột 4 (index 3) là CPOSTAG/UPOS Gold.
            poses = [row[3] for row in sentence_rows]
            if any(not pos or pos == "_" for pos in poses):
                raise ValueError(
                    f"Câu {index}: thiếu UPOS Gold trong cột 4 của {test_path}."
                )

            if len(tokens) != len(poses):
                raise ValueError(
                    f"Câu {index}: có {len(tokens)} token "
                    f"nhưng {len(poses)} nhãn POS."
                )

            # Ghi UPOS Gold vào file đầu vào MaltParser.
            # HEAD/DEPREL vẫn là placeholder để MaltParser dự đoán.
            write_conll_sentence(
                out,
                f"test_{index}",
                tokens,
                poses
            )

            total_tokens += len(tokens)

    print(f"Số câu: {len(sentences)}")
    print(f"Số token: {total_tokens}")
    print(f"Đã tạo đầu vào MaltParser: {output_path}")


if __name__ == "__main__":
    main()