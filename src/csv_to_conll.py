"""
Chuyển dữ liệu CSV có nhãn dependency sang định dạng CoNLL-X cho MaltParser.
Maltparser sẽ học từ gold của pos trước khi dự đoán dựa trên dự đoán từ mô hình pos dùng crf.
LƯU Ý:
- CoNLL-X có 10 cột, phân cách bằng tab:  ID, FORM, LEMMA, CPOSTAG, POSTAG, FEATS, HEAD, DEPREL, PHEAD, PDEPREL.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
import pandas as pd
import re
import ast

# Cho phép import các tệp Python nằm cùng thư mục với file hiện tại.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pos_tagging import split_dataset
from main import prepared_df


# Những cột này phải chứa danh sách theo từng token của một câu.
REQUIRED_COLUMNS = ["tokens", "upos", "upos","feats", "head", "deprel"]


#phải chyển thành python do pandas đọc xong thì các cột có danh sách thật ra đang ở dạng chuỗi rồi
def parse_list_cell(value):
    """Đọc một ô CSV có dạng danh sách JSON thành list trong python."""
    #kiểm tra xem phải kiểu danh sách chưa
    if isinstance(value, list):
        return value
    #trống thì trả lại danh sách rỗng
    if value is None:
        return []
    # Xử lý giá trị NaN do pandas đọc từ CSV.
    if pd.isna(value):
        return []
    #chuyển thành chuỗi và bỏ khoảng trắng ở đầu cuối chuỗi
    value = str(value).strip()
    if not value:
        return []
    #thử chyển chuỗi thành dữ liệu python
    try:
        #đọc ở dạng json và chuyển thành đối tượng trong python
        parsed = json.loads(value)
        #coi chính xác có phải là danh sách  không, nếu không có thể là chuỗi đơn hoặc số
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Dạng numpy ['a' 'b'] (không có dấu phẩy): literal_eval sẽ ghép thành 'ab' nên bỏ qua bước này
    numpy_style = re.search(r"'\s+'|\"\s+\"", value) is not None
    if not numpy_style:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
                pass
    
    # Tách các phần tử nằm trong dấu nháy đơn hoặc nháy kép
    # ['Thời hạn' 'hưởng' 'quyền' '.']
    items = re.findall(r"'((?:\\.|[^'])*)'|\"((?:\\.|[^\"])*)\"", value)
    if items:
        return [(a if a else b).replace("\\'", "'").replace('\\"', '"') for a, b in items]
    
    # Hỗ trợ chuỗi danh sách Python, ví dụ: ['Tôi', 'đi'].
    try:
        import ast
        #chạy thêm kiểm tra, tại json yêu cầu chuỗi có ngoặc kép mà python có thể trả theo cú pháp danh sách ""
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def conll_block(tokens, upos, feats, heads, deprels):
    """Chuyển một câu thành các dòng CoNLL-X gồm đúng 10 cột."""
    lines = []

    for i, (token, up, feat, head, deprel) in enumerate(
        zip(tokens, upos, feats, heads, deprels), start=1
    ):
        token = str(token).strip().replace("\t", " ").replace("\n", " ") or "_"
        lemma = token.lower().replace(" ", "_")

        columns = [
            str(i),                       # 1. ID
            token,                         # 2. FORM
            lemma,                         # 3. LEMMA
            str(up).strip() or "_",        # 4. CPOSTAG: Gold UPOS
            str(up).strip() or "_",        # 5. POSTAG: Gold UPOS
            str(feat).strip() or "_",      # 6. FEATS
            str(head),                     # 7. HEAD
            str(deprel).strip() or "_",    # 8. DEPREL
            "_",                           # 9. PHEAD
            "_",                           # 10. PDEPREL
        ]

        lines.append("\t".join(columns))

    return "\n".join(lines) + "\n\n"

#nhận vào một hàng trong df và số dòng của câu trong df nguồn để truy vết lỗi
def validate_sentence(row, source_row_number):
    """Kiểm tra các cột của một câu trước khi ghi vào CoNLL-X."""
    #đi qua từng cột trong REQUỈED_COLUMNS để lấy giá trị tương ứng từ row rồi dùng hàm parse để chuyển dữ liệu thành danh sách Python
    lists = {
        column: parse_list_cell(row.get(column))
        for column in REQUIRED_COLUMNS
    }
    tokens = lists["tokens"]
    if not tokens:
        return None, "Không có token hoặc không đọc được cột tokens"

    token_count = len(tokens)
    # mỗi cột phải có cùng số token.
    for column, values in lists.items():
        if len(values) != token_count:
            return None, (f"Cột {column} có {len(values)} phần tử, nhưng tokens có {token_count} phần tử")

    # HEAD phải là số nguyên và nằm trong khoảng 0..số token.
    try:
        heads = [int(str(value).strip()) for value in lists["head"]]
    except (TypeError, ValueError):
        return None, "Cột head có giá trị không phải số nguyên"
    #coi có mấy từ có head là 0 và trong 1 câu chỉ được 1 từ có head là 0 thôi
    if sum(head == 0 for head in heads) != 1:
        return None, "Câu có nhiều hơn một token có HEAD = 0"
    #kiểm tra head có lấy ngoài phạm vi không, 3 từ thì head có 2, 2 từ thì head có 1 chứ không được 5,6
    if any(head < 0 or head > token_count for head in heads):
        return None, "Cột head có tham chiếu ngoài phạm vi token"
    #tự làm cha của nó cũng không được
    if any(head == index for index, head in enumerate(heads, start=1)):
        return None, "Có token tự làm HEAD của chính nó"
    if any(not str(label).strip() for label in lists["deprel"]):
        return None, "Cột deprel có nhãn rỗng"
    # Bảo đảm các nhãn POS không bị rỗng trước khi xuất.
    if any(not str(label).strip() for label in lists["upos"]):
        return None, "Cột upos có nhãn rỗng"
    #xuất dict cho câu đó
    sentence = {
        "source_row": source_row_number,
        "tokens": [str(value) for value in tokens],
        "upos": [str(value) for value in lists["upos"]],
        "upos": [str(value) for value in lists["upos"]],
        "feats": [str(value) if str(value).strip() else "_" for value in lists["feats"]],
        "heads": heads,
        "deprels": [str(value) for value in lists["deprel"]],
    }
    return sentence, None

#ghi các câu đã kiểm rta vào tệp conll-X sau khi chuyển python và kiểm tra đầy đủ
#đầu vào gồm đương dẫn file conll-X sẽ ghi ra, df chứa các câu cần xử lý, tập các câu lỗi trong quá trnifh chuyển sang conll-X để biest chính xác câu nào bị bỏ qua và danh sách dùng để lưu thông tin các dòng bị bỏ qua
def write_conll(path, rows, split_name, skipped_rows):
    """Ghi các câu hợp lệ của một tập ra tệp CoNLL-X."""
    sentence_count = 0
    #mở file để ghi
    with open(path, "w", encoding="utf-8", newline="\n") as output_file:
        #duyệt từng dòng trong df
        for original_index, (_, row) in enumerate(rows.iterrows()):
            # Số dòng nguồn tính cả dòng tiêu đề CSV là chỉ số DataFrame + 2.
            source_row_number = int(original_index) + 2
            #kiểm tra hợp lệ
            sentence, error = validate_sentence(row, source_row_number)

            #lưu các dòng lỗi bị bỏ qua
            if error:
                skipped_rows.append({
                    "split": split_name, #thuộc tập train hay test hay valid
                    "csv_row": source_row_number, #dòng mấy
                    "reason": error,#tại sao bị lỗi
                })
                continue

            output_file.write(conll_block(
                sentence["tokens"],
                sentence["upos"],
                sentence["feats"],
                sentence["heads"],
                sentence["deprels"],
            ))
            sentence_count += 1

    return sentence_count
#trả về số câu được ghi thành công

#xuất dữ liệu của tập test thành một file csv để chuẩn bị đầu vào cho malt
def write_pos_input(path, rows, skipped_rows):
    """
    Xuất câu thuộc tập test làm đầu vào cho MaltParser.
    Tệp này không chứa HEAD/DEPREL để tránh đưa đáp án chuẩn vào đầu vào dự đoán.
    """
    fieldnames = ["source_row", "text", "tokens", "upos"]

    with open(path, "w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for original_index, (_, row) in enumerate(rows.iterrows()):
            source_row_number = int(original_index) + 2
            sentence, error = validate_sentence(row, source_row_number)

            if error:
                # Nếu câu đã lỗi ở test.conll thì không đưa câu đó vào đầu vào dự đoán.
                # Lỗi cũng được ghi nhận ở skipped_rows.csv.
                skipped_rows.append({
                    "split": "test_input",
                    "csv_row": source_row_number,
                    "reason": error,
                })
                continue

            text_value = row.get("text", "")
            if pd.isna(text_value):
                text_value = ""

            writer.writerow({
                "source_row": source_row_number,
                "text": str(text_value),
                "tokens": json.dumps(sentence["tokens"], ensure_ascii=False),
                "upos": json.dumps(sentence["upos"], ensure_ascii=False),
            })


def main():
    parser = argparse.ArgumentParser(description="Chuyển CSV thành CoNLL-X.")
    parser.add_argument("--outdir", default="data", help="Thư mục lưu các tệp đầu ra")
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepared_df()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "CSV thiếu các cột bắt buộc: " + ", ".join(missing_columns)
        )

    # Chia bằng đúng hàm dùng trong POS. Hàm này cần trả về DataFrame gốc
    # để giữ lại HEAD/DEPREL và thông tin định danh của từng câu.
    test_df, valid_df, train_df = split_dataset(df)

    # Kiểm tra để phát hiện trường hợp hàm chia tập làm mất hoặc lặp dòng.
    total_split_rows = len(train_df) + len(valid_df) + len(test_df)
    if total_split_rows != len(df):
        raise ValueError(
            "Số dòng sau khi chia tập không bằng số dòng CSV ban đầu. "
            "Hãy kiểm tra split_dataset() trong pos_tagging.py."
        )
    # Ghi các tập có nhãn chuẩn
    skipped_rows = [] #ghi lại các câu bị bỏ luôn
    write_conll(output_dir / "train.conll", train_df, "train", skipped_rows)
    write_conll(output_dir / "valid.conll", valid_df, "valid", skipped_rows)
    write_conll(output_dir / "test.conll", test_df, "test", skipped_rows)
    print(f"Đã lưu các tệp vào: {output_dir.resolve()}")
#trả về train.coll, test.conll, valid.conll và malt