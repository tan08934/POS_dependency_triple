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
from pos_tagging import features, features2, sentfeatures, train_test_valid, train_crf, predict, evaluate_pos, make_features, error_analysis
from rule_extractor import TokenNode, extract_triples_openie
from evaluator import evaluate_predictions
from validate_dependency import read_conll


# =====================================================================
# BƯỚC 1 - POS: toàn bộ code huấn luyện/đánh giá CRF cũ, bọc vào hàm
# (trước đây chạy ngay khi import nên không gọi main() từ giao diện được)
# =====================================================================
def train_pos_model():
    df = load_data()
    print("=========CHUẨN HÓA ============")
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

    print("========DATA LEAKGAE KIỂM TRA TRÙNG VÀ CỘT RỖNG===========")
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

    print("========SO SÁNH HAI MÔ HÌNH CỬA SỐ +-1 VÀ +-2============")
    #huấn luyện CRF
    #mô hình với cửa sổ là +-1
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid(df,features,False,False)
    print("Số câu train:", len(X_train), len(y_train))

    crf = train_crf(X_train,y_train)
    y_pred = predict(crf,X_test)

    #mô hình có thêm đặc trưng cửa sổ là +-2
    X_train2, y_train2, X_valid2, y_valid2, X_test2, y_test2 = train_test_valid(df,features2,False,False)
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
    
    print("==========PHẦN CẢI TIẾN FEATURES===========")
    #cải tiến
    #huấn luyện CRF
    #mô hình với cửa sổ là +-1 (gốc chưa có cải tiến)
    X_train1, y_train1, X_valid1, y_valid1, X_test1, y_test1 = train_test_valid(df,features,False,False)
    crf1 = train_crf(X_train1,y_train1)
    y_pred1 = predict(crf1,X_test1)
    
    #huấn luyện CRF
    #mô hình với cửa sổ là +-1 (có thêm từ láy và từ chức năng)
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid(df,features,True,True)
    crf = train_crf(X_train,y_train)
    y_pred = predict(crf,X_test)
    
    #đánh giá hai mô hình trước và sau khi có cải tiến đặc trưng
    s_old = evaluate_pos(y_test1,y_pred1)
    s = evaluate_pos(y_test,y_pred)
    
    df_old = lib.pd.DataFrame(s_old).T
    df_new = lib.pd.DataFrame(s).T
    #đổi tên cột để phân biệt hai mô hình
    df_old = df_old.add_suffix("_old")
    df_new = df_new.add_suffix("_new")
    # Ghép kết quả theo nhãn POS
    comparison = lib.pd.concat([df_old, df_new], axis=1)
    print(comparison.round(2))

    #so sánh 
    f1_com = comparison[["f1_old", "f1_new"]].copy()
    f1_com["Chênh lệch f1"] = (f1_com["f1_new"] - f1_com["f1_old"])
    print(f1_com.round(2).to_string())

    ac_com = comparison[["accuracy_old", "accuracy_new"]].copy()
    ac_com["Chênh lệch accuracy"] = (ac_com["accuracy_new"] - ac_com["accuracy_old"])
    print(ac_com.round(2).to_string())

    pre_com = comparison[["precision_old", "precision_new"]].copy()
    pre_com["Chênh lệch precision"] = (pre_com["precision_new"] - pre_com["precision_old"])
    print(pre_com.round(2).to_string())

    re_com = comparison[["recall_old", "recall_new"]].copy()
    re_com["Chênh lệch recall"] = (re_com["recall_new"] - re_com["recall_old"])
    print(re_com.round(2).to_string())

    # Sau khi huấn luyện xong
    MODELS_DIR.mkdir(exist_ok=True)
    lib.joblib.dump(crf, POS_MODEL)

    print("========CÁC LỖI PHỔ BIẾN SAU KHI POS ========")
    print("KIỂM TRA TRÊN MÔ HÌNH CHƯA CẢI TIẾN FEATURES")
    error_df1 = error_analysis(X_test1, y_test1, y_pred1)
    
    print("\n KIỂM TRA TRÊN MÔ HÌNH ĐÃ CẢI TIẾN FEATURES")
    error_df = error_analysis(X_test,y_test,y_pred)
    
    print("======KIỂM TRA ĐỘ QUAN TRỌNG CỦA TỪNG FEATURES VỚI MÔ HÌNH DỰ ĐOÁN CRF========")
    #không chuẩn hóa chữ thường 
    print("Trường hợp không chuẩn hóa từ về chữ thường.")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['token.lower']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")

    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
    print("Trường hợp không kiểm tra viết hoa cả từ")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['is_upper']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")
    
    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
    print("Trường hợp không kiểm tra chữ đầu")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['is_title']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")
    
    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
    print("Trường hợp không kiểm tra là số")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['is_digit']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")
    
    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
    print("Trường hợp không kiểm tra features là dấu cấu hay không")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['is_punct']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")
    
    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
    print("Trường hợp không có features độ dài từ")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['length']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")
    
    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
    print("Trường hợp không có ngữ cảnh trước và sau")
    X_train_t, y_train_t, X_valid_t, y_valid_t, X_test_t, y_test_t = train_test_valid(df,make_features(['prev.lower','next.lower']),True,True) 
    crf_t = train_crf(X_train_t,y_train_t)
    y_pred_t = predict(crf_t,X_test_t)

    s_t = evaluate_pos(y_test_t,y_pred_t)
    df_t = lib.pd.DataFrame(s_t).T
    df_t = df_t.add_suffix("_t")
    
    comparison_t = lib.pd.concat([df_t, df_new], axis=1)
    f1_t = comparison_t[["f1_t", "f1_new"]].copy()
    f1_t["Chênh lệch f1"] = (f1_t["f1_new"] - f1_t["f1_t"])
    print(f1_t.round(2).to_string())

    ac_t = comparison_t[["accuracy_t", "accuracy_new"]].copy()
    ac_t["Chênh lệch accuracy"] = (ac_t["accuracy_new"] - ac_t["accuracy_t"])
    print(ac_t.round(2).to_string())

    pre_t = comparison_t[["precision_t", "precision_new"]].copy()
    pre_t["Chênh lệch precision"] = (pre_t["precision_new"] - pre_t["precision_t"])
    print(pre_t.round(2).to_string())

    re_t = comparison_t[["recall_t", "recall_new"]].copy()
    re_t["Chênh lệch recall"] = (re_t["recall_new"] - re_t["recall_t"])
    print(re_t.round(2).to_string())
    #error_t = error_analysis(X_test_t,y_test_t,y_pred_t)
    
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


def evaluate_triples(crf=None, gold_path=None, match_mode="exact"):
    """Chạy pipeline trên data/gold_annotated.json (hoặc dataset_test.json) rồi chấm bằng evaluator.py."""
    crf = crf or load_pos_model()
    
    if gold_path is None:
        gold_file = DATA_DIR / "gold_annotated.json"
        if not gold_file.exists():
            gold_file = DATA_DIR / "dataset_test.json"
    else:
        gold_file = Path(gold_path)

    if not gold_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gold tại {gold_file}")

    print(f"📖 Đang đọc tập dữ liệu kiểm thử từ: {gold_file}")
    with gold_file.open(encoding="utf-8") as f:
        dataset = json.load(f)

    tagged = [tag_sentence(item.get("sentence", item.get("text", "")), crf) for item in dataset]
    parsed = parse_dependencies(tagged)
    results = [{
        "sentence": item.get("sentence", item.get("text", "")),
        "type": item.get("type", ""),
        "gold_triples": item.get("gold_triples", []),
        "predicted_triples": extract_triples_openie(nodes),
    } for item, nodes in zip(dataset, parsed)]

    report = evaluate_predictions(results, match_mode=match_mode)
    out = DATA_DIR / "triple_evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "error_logs"}, ensure_ascii=False, indent=2))
    print("Đã lưu kết quả tại:", out.resolve())
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
