"""
Pipeline: chuẩn hóa -> tách từ -> POS (CRF) -> dependency (MaltParser) -> bộ ba (rule_extractor).

"""
import sys
import json
import argparse
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))  # để import được các module phẳng trong src/ dù chạy từ đâu

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
TOOLS_DIR = ROOT_DIR / "tools"
MALT_JAR = TOOLS_DIR / "maltparser-1.9.2" / "maltparser-1.9.2.jar"
MALT_ZIP = TOOLS_DIR / "maltparser-1.9.2.zip"
MALT_MODEL = MODELS_DIR / "vietnamese.mco"
POS_MODEL = MODELS_DIR / "pos_crf.pkl"

from preprocessing import normalize_text, tokenize, duplicate, del_duplicate, emptycol, del_emptycol, preprocess_data
import lib
from data_loader import load_data
from word_segmetation.segment import segment_text
from pos_tagging import features, features2, sentfeatures, train_test_valid, train_crf, predict, evaluate_pos
from rule_extractor import TokenNode, extract_triples_openie
from evaluator import evaluate_predictions
from validate_dependency import read_conll


# =====================================================================
# BƯỚC 1 - POS: toàn bộ code huấn luyện/đánh giá CRF cũ, bọc vào hàm
# (trước đây chạy ngay khi import nên không gọi main() từ giao diện được)
# =====================================================================
def train_pos_model():
    df = load_data()
    #chuẩn hóa
    df["normalize"] = df["text"].map(normalize_text)
    print(df["normalize"])

    #tokenization
    tokenized_corpus = [tokenize(s) for s in df["text"]]
    word_freq = lib.Counter()
    for sent in tokenized_corpus:
        word_freq.update(sent)
    vocab = set(word_freq) # chỉ lấy các từ vựng không lấy số lần xuất hiện
    print("Số câu:", len(tokenized_corpus))
    print("Vocabulary size:", len(vocab))
    print("10 từ phổ biến:", word_freq.most_common(10))

    #kiểm tra trùng lặp
    print("Số dòng ban đầu:",len(df))
    dup1 = duplicate(df["text"]) # được một Series có giá trị Bool
    df = del_duplicate(df,"text")
    #kiểm tra lại lần nữa còn trùng không
    print("Kiểm tra trùng lặp sau khi đã xử lý lần 1.")
    print(duplicate(df["text"]).sum())
    print("Đánh lại số dòng thành công, số dòng sau khi loại bỏ trùng và rỗng:",len(df))

    # có cột rỗng không
    empty  = emptycol(df)
    df = del_emptycol(df,empty)
    print(df["upos"])
    print(type(df.loc[0, "upos"]))
    print(len(df.loc[0, "upos"]))

    #tách từ
    df["word_segmentation"] = df["text"].map(segment_text)
    lib.display(df.head(10))


    #huấn luyện CRF
    #mô hình với cửa sổ là +-1
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid(df,features)
    print("Số câu train:", len(X_train), len(y_train))

    print("X:", len(X_train[0]))
    print("y:", len(y_train[0]))

    print("word_segmentation:", df["word_segmentation"].iloc[0])
    print("upos:", df["upos"].iloc[0])

    crf = train_crf(X_train,y_train)
    y_pred = predict(crf,X_test)

    #mô hình có thêm đặc trưng cửa sổ là +-2
    X_train2, y_train2, X_valid2, y_valid2, X_test2, y_test2 = train_test_valid(df,features2)
    crf2 = train_crf(X_train2,y_train2)
    y_pred2 = predict(crf2,X_test2)

    #kiểm tra xem
    print("Cùng nhãn test:", y_test == y_test2)

    #gọi hàm đánh giá để xem mô hình nào tốt hơn để lấy
    s1 = evaluate_pos(y_test,y_pred)
    s2 = evaluate_pos(y_test2,y_pred2)
    #biểu diễn để dễ so sánh
    #chuyển dictionary thành DataFrame
    #xoay hàng cột
    df1 = lib.pd.DataFrame(s1).T
    df2 = lib.pd.DataFrame(s2).T
    #đổi tên cột để phân biệt hai mô hình
    df1 = df1.add_suffix("_window1")
    df2 = df2.add_suffix("_window2")
    # Ghép kết quả theo nhãn POS
    comparison = lib.pd.concat([df1, df2], axis=1)
    print(comparison.round(2))

    #so sánh 
    f1_compare = comparison[["f1_window1", "f1_window2"]].copy()
    f1_compare["Chênh lệch f1"] = (f1_compare["f1_window2"] - f1_compare["f1_window1"])
    print(f1_compare.round(2).to_string())

    ac_compare = comparison[["accuracy_window1", "accuracy_window2"]].copy()
    ac_compare["Chênh lệch accuracy"] = (ac_compare["accuracy_window2"] - ac_compare["accuracy_window1"])
    print(ac_compare.round(2).to_string())

    pre_compare = comparison[["precision_window1", "precision_window2"]].copy()
    pre_compare["Chênh lệch precision"] = (pre_compare["precision_window2"] - pre_compare["precision_window1"])
    print(pre_compare.round(2).to_string())

    re_compare = comparison[["recall_window1", "recall_window2"]].copy()
    re_compare["Chênh lệch recall"] = (re_compare["recall_window2"] - re_compare["recall_window1"])
    print(re_compare.round(2).to_string())

    lib.display(df1[["accuracy_window1","precision_window1","recall_window1","f1_window1"]])

    # Sau khi huấn luyện xong
    MODELS_DIR.mkdir(exist_ok=True)
    lib.joblib.dump(crf, POS_MODEL)

    # Khi chạy giao diện thì mở code này
    #crf = lib.joblib.load("models/pos_crf.pkl")

    #Xem các lỗ phổ biến:
    errors = []

    for sent_idx, (X_sent, y_true_sent, y_pred_sent) in enumerate(
        zip(X_test, y_test, y_pred)
    ):
        for token_idx, (feat, gold, pred) in enumerate(
            zip(X_sent, y_true_sent, y_pred_sent)
        ):
            if gold == pred:
                continue

            token = feat.get("token.lower", "")
            prev = feat.get("prev.lower", "")
            next_ = feat.get("next.lower", "")

            # Nhận diện dạng dữ liệu
            #có phải dạng số hoặc ngày tháng không
            is_number_like = bool(
                lib.re.fullmatch(r"[\d.,/%\-]+", token)
            )
            #là viết tắt có chữ cái viết hoa và nhiều hơn 1 ký tự
            is_viettat = (
                token.isupper() and len(token) > 1
            )
            is_multiword = " " in token

            # Gắn các nhóm có thể khảo sát
            categories = []
            if is_number_like:
                categories.append("Dạng số")
            if is_viettat:
                categories.append("Chữ viết tắt")
            if is_multiword:
                categories.append("Cụm nhiều từ")
            if token_idx == 0:
                categories.append("Đầu câu")
            if token_idx == len(X_sent) - 1:
                categories.append("Cuối câu")

            errors.append({
                "sentence_index": sent_idx,
                "token_index": token_idx,
                "token": token,
                "gold": gold,
                "predicted": pred,
                "prev": prev,
                "next": next_,
                "error_pair": f"{gold} -> {pred}",
                "categories": categories
            })
    error_df = lib.pd.DataFrame(errors)
    #thống kê các lỗi phổ biến

    #Trường hợp những cặp nhãn bị nhầm nhiều nhất
    print("Các cặp nhãn bị nhầm nhiều nhất:")
    lib.display(error_df["error_pair"].value_counts().head(10))
    #Trường hợp các từ bị nhầm nhiều nhất
    print("Các token bị nhầm nhiều nhất:")
    lib.display(error_df["token"].value_counts().head(10))
    #trường hợp các nhóm lỗi được gắn nhãn theo hình thức/vị trí
    category_counts = lib.Counter(
        category
        for categories in error_df["categories"]
        for category in categories
    )
    print("Số lỗi thuộc từng nhóm khảo sát:")
    lib.display(lib.pd.DataFrame(
        category_counts.items(),
        columns=["Nhóm lỗi", "Số token lỗi"]
    ).sort_values("Số token lỗi", ascending=False))
    print("Tổng số token dự đoán sai:", len(error_df))
    lib.display(error_df)

    return crf


def load_pos_model():
    if not POS_MODEL.exists():
        raise FileNotFoundError(f"Chưa có {POS_MODEL}. Chạy train_pos_model() hoặc bỏ cờ --skip-pos-train.")
    return lib.joblib.load(POS_MODEL)


# =====================================================================
# BƯỚC 2 - Dependency theo lô: các script README "Bước 1-3" chạy lần lượt
# =====================================================================
def rel(path):
    """Đường dẫn tương đối so với thư mục gốc (các script chạy với cwd=ROOT_DIR như README)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def run_script(script, *args):
    cmd = [sys.executable, str(SRC_DIR / script), *[str(a) for a in args]]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def find_udd_csv():
    for p in (SRC_DIR / "UDD-1.csv", DATA_DIR / "UDD-1.csv", ROOT_DIR / "UDD-1.csv"):
        if p.exists():
            return p
    raise FileNotFoundError("Không tìm thấy UDD-1.csv (đã tìm trong src/, data/ và thư mục gốc).")


def ensure_maltparser():
    if MALT_JAR.exists():
        return
    if MALT_ZIP.exists():
        print("Giải nén", MALT_ZIP)
        with zipfile.ZipFile(MALT_ZIP) as z:
            z.extractall(TOOLS_DIR)
    if not MALT_JAR.exists():
        raise FileNotFoundError(f"Không tìm thấy {MALT_JAR}. Xem README mục 3.3 để tải MaltParser.")


def run_dependency_pipeline(retrain=False):
    ensure_maltparser()
    jar = rel(MALT_JAR)

    # Bước 1: UDD-1.csv -> train.conll, test.conll, input_pos.csv
    run_script("csv_to_conll.py", "--input", rel(find_udd_csv()), "--outdir", "data")

    # Bước 2: huấn luyện MaltParser (bỏ qua nếu đã có model)
    if retrain or not MALT_MODEL.exists():
        run_script("train_maltparser.py", "--jar", jar, "--train", "data/train.conll")
    else:
        print(f"Đã có {rel(MALT_MODEL)}, bỏ qua huấn luyện (dùng --retrain-malt để train lại).")

    # Bước 3: input -> dự đoán -> kiểm tra cấu trúc -> đánh giá UAS/LAS
    run_script("prepare_malt_input.py", "--input", "data/input_pos.csv", "--output", "data/malt_input.conll")
    run_script("run_maltparser.py", "--jar", jar, "--model", rel(MALT_MODEL),
               "--input", "data/malt_input.conll", "--output", "data/dependency_predictions.conll")
    run_script("validate_dependency.py", "--input", "data/dependency_predictions.conll")
    run_script("evaluate_dependency.py", "--gold", "data/test.conll", "--pred", "data/dependency_predictions.conll")


# =====================================================================
# BƯỚC 3 - Dùng pipeline cho câu mới: POS -> MaltParser -> bộ ba
# =====================================================================
_UPOS2XPOS = None


def upos_to_xpos():
    """MaltParser được train với CPOSTAG=upos, POSTAG=xpos (train.conll).
    CRF chỉ dự đoán upos nên lấy xpos phổ biến nhất của mỗi upos trong train.conll."""
    global _UPOS2XPOS
    if _UPOS2XPOS is None:
        counts = {}
        train = DATA_DIR / "train.conll"
        if train.exists():
            with train.open(encoding="utf-8") as f:
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) >= 5:
                        counts.setdefault(cols[3], Counter())[cols[4]] += 1
        _UPOS2XPOS = {u: c.most_common(1)[0][0] for u, c in counts.items()}
    return _UPOS2XPOS


def tag_sentence(text, crf):
    """Tách từ + gán POS (cùng cách tách từ như lúc huấn luyện CRF: segment_text trên câu gốc)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Vui lòng nhập vào một câu tiếng Việt.")
    text = preprocess_data(text)
    tokens = segment_text(text)
    X = sentfeatures(tokens, features)
    return tokens, predict(crf, [X])[0]
#do predict nhiều câu, lấy đúng câu mình thả vào thôi


def parse_dependencies(sentences):
    """sentences: list[(tokens, upos)] -> list[list[TokenNode]]. Chỉ gọi MaltParser một lần cho cả lô."""
    ensure_maltparser()
    mapping = upos_to_xpos()
    DATA_DIR.mkdir(exist_ok=True)
    tmp_in, tmp_out = DATA_DIR / "_tmp_malt_input.conll", DATA_DIR / "_tmp_malt_output.conll"
    with tmp_in.open("w", encoding="utf-8", newline="\n") as f:
        for tokens, upos in sentences:
            for i, (tok, up) in enumerate(zip(tokens, upos), start=1):
                tok = tok.strip().replace("\t", " ") or "_"
                f.write("\t".join([str(i), tok, tok.lower().replace(" ", "_"), up,
                                   mapping.get(up, up), "_", "0", "root", "_", "_"]) + "\n")
            f.write("\n")
    run_script("run_maltparser.py", "--jar", rel(MALT_JAR), "--model", rel(MALT_MODEL),
               "--input", rel(tmp_in), "--output", rel(tmp_out))
    parsed = read_conll(tmp_out)
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    if len(parsed) != len(sentences):
        raise RuntimeError(f"MaltParser trả {len(parsed)} câu, đầu vào {len(sentences)} câu.")
    return [[TokenNode(int(r[0]), r[1], int(r[6]), r[7], pos=up) for r, up in zip(rows, upos)]
            for rows, (_, upos) in zip(parsed, sentences)]


def main(text, crf=None):
    """Chạy cả pipeline cho một câu (dùng được trực tiếp từ giao diện)."""
    crf = crf or load_pos_model()

    # Bước 1-2: tách từ + POS
    tokens, pos_result = tag_sentence(text, crf)

    # Bước 3: dependency parsing (MaltParser)
    nodes = parse_dependencies([(tokens, pos_result)])[0]

    # Bước 4: trích bộ ba (subject, relation, object)
    triples = extract_triples_openie(nodes)

    return {
        "text": text,
        "tokens": tokens,
        "pos": pos_result,
        "dependency": [{"id": n.id, "token": n.form, "pos": n.pos, "head": n.head, "deprel": n.deprel}
                       for n in nodes],
        "triples": triples,
    }


def evaluate_triples(crf, match_mode="exact"):
    """Chạy pipeline trên data/dataset_test.json rồi chấm bằng evaluator.py."""
    with (DATA_DIR / "dataset_test.json").open(encoding="utf-8") as f:
        dataset = json.load(f)
    tagged = [tag_sentence(item["sentence"], crf) for item in dataset]
    parsed = parse_dependencies(tagged)
    results = [{
        "sentence": item["sentence"],
        "type": item.get("type", ""),
        "gold_triples": item["gold_triples"],
        "predicted_triples": extract_triples_openie(nodes),
    } for item, nodes in zip(dataset, parsed)]
    report = evaluate_predictions(results, match_mode=match_mode)
    out = DATA_DIR / "triple_evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "error_logs"}, ensure_ascii=False, indent=2))
    print("Đã lưu:", out.resolve())
    return report


# =====================================================================
# Chạy tất cả một lượt
# =====================================================================
def run_all(skip_pos_train=False, retrain_malt=False, demo_text="Chính phủ ban hành quy định mới."):
    print("\n===== [1/4] POS tagging (CRF) =====")
    crf = load_pos_model() if skip_pos_train else train_pos_model()

    print("\n===== [2/4] Dependency parsing: csv_to_conll -> train -> prepare -> run -> validate -> evaluate =====")
    run_dependency_pipeline(retrain=retrain_malt)

    print("\n===== [3/4] Trích bộ ba + đánh giá trên dataset_test.json =====")
    evaluate_triples(crf)

    print("\n===== [4/4] Demo một câu =====")
    result = main(demo_text, crf)
    print(json.dumps(result, ensure_ascii=False, indent=2,default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chạy toàn bộ pipeline POS -> dependency -> triple.")
    ap.add_argument("--skip-pos-train", action="store_true", help="dùng models/pos_crf.pkl, không train lại CRF")
    ap.add_argument("--retrain-malt", action="store_true", help="train lại MaltParser dù đã có vietnamese.mco")
    ap.add_argument("--text", default="Chính phủ ban hành quy định mới.", help="câu demo ở bước cuối")
    cli = ap.parse_args()
    run_all(cli.skip_pos_train, cli.retrain_malt, cli.text)
